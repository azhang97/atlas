class MetaUNetATLASModel(ATLASModel):
  def __init__(self, FLAGS):
    """
    Initializes the ATLAS model.

    Inputs:
    - FLAGS: A _FlagValuesWrapper object.
    """
    self.FLAGS = FLAGS

    with tf.variable_scope("ATLASModel"):
      self.add_placeholders()
      self.build_graph()
      self.add_loss()

    # Defines the trainable parameters, gradient, gradient norm, and clip by
    # gradient norm
    params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, params)
    self.gradient_norm = tf.global_norm(gradients)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                  FLAGS.max_gradient_norm)
    self.param_norm = tf.global_norm(params)

    # Defines optimizer and updates; {self.updates} needs to be fetched in
    # sess.run to do a gradient update
    self.global_step_op = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step=self.global_step_op)

    # Adds a summary to write examples of images to TensorBoard
    utils.add_summary_image_triplet(self.inputs_op[:,:,0],
                                    self.target_masks_op,
                                    self.predicted_masks_op,
                                    num_images=self.FLAGS.num_summary_images)

    # Defines savers (for checkpointing) and summaries (for tensorboard)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
    self.summaries = tf.summary.merge_all()

  def add_placeholders(self):
    """
    Adds placeholders to the graph.

    Defines:
    - self.batch_size_op: A scalar placeholder Tensor that represents the
      batch size.
    - self.inputs_op: A placeholder Tensor with shape batch size by image dims
      e.g. (100, 233, 197) that represents the batch of inputs.
    - self.target_masks_op: A placeholder Tensor with shape batch size by mask
      dims e.g. (100, 233, 197) that represents the batch of target masks.
    - self.keep_prob: A scalar placeholder Tensor that represents the keep
      probability for dropout.
    """
    # Adds placeholders for inputs
    self.batch_size_op = tf.placeholder(tf.int32, shape=(), name="batch_size")

    # Defines the input dimensions, which depend on the intended input; here
    # the intended input is a single slice but volumetric inputs might require
    # 1+ additional dimensions
    self.input_dims = [self.FLAGS.slice_height, self.FLAGS.slice_width, 5]
    self.mask_dims = [self.FLAGS.slice_height, self.FLAGS.slice_width]
    self.output_dims = self.mask_dims

    # Defines input and target segmentation mask according to the input dims
    self.inputs_op = tf.placeholder(tf.float32,
                                    shape=[None] + self.input_dims,
                                    name="input")
    self.target_masks_op = tf.placeholder(tf.float32,
                                          shape=[None] + self.mask_dims,
                                          name="target_mask")

    # Adds a placeholder to feed in the keep probability (for dropout)
    self.keep_prob = tf.placeholder_with_default(1.0, shape=())

  def build_graph(self):
    metaunet = MetaUNet(input_shape=self.input_dims,
                    keep_prob=self.keep_prob,
                    output_shape=self.input_dims,
                    scope_name="unet")
    self.logits_op = tf.squeeze(
      metaunet.build_graph(self.inputs_op))

    self.predicted_mask_probs_op = tf.sigmoid(self.logits_op,
                                              name="predicted_mask_probs")
    self.predicted_masks_op = tf.cast(self.predicted_mask_probs_op > 0.5,
                                      tf.uint8,
                                      name="predicted_masks")

  def train(self,
            sess,
            train_input_paths,
            train_target_mask_paths,
            dev_input_paths,
            dev_target_mask_paths):
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))

    # We will keep track of exponentially-smoothed loss
    exp_loss = None

    # Checkpoint management.
    # We keep one latest checkpoint, and one best checkpoint (early stopping)
    checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
    best_dev_dice_coefficient = None

    # For TensorBoard
    summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, sess.graph)

    epoch = 0
    num_epochs = self.FLAGS.num_epochs
    while num_epochs == None or epoch < num_epochs:
      epoch += 1

      # Loops over batches
      msbg = MetaSliceBatchGenerator(train_input_paths,
                                     train_target_mask_paths,
                                     self.FLAGS.batch_size,
                                     1000, None, (self.FLAGS.slice_height,
                                                  self.FLAGS.slice_width),
                                     False, self.FLAGS.use_fake_target_masks)
      num_epochs_str = str(num_epochs) if num_epochs != None else "indefinite"
      for batch in tqdm(msbg.get_batch(),
                        desc=f"Epoch {epoch}/{num_epochs_str}",
                        total=msbg.get_num_batches()):
        # Runs training iteration
        loss, global_step, param_norm, grad_norm =\
          self.run_train_iter(sess, batch, summary_writer)

        # Updates exponentially-smoothed loss
        if not exp_loss:  # first iter
          exp_loss = loss
        else:
          exp_loss = 0.99 * exp_loss + 0.01 * loss

        # Sometimes prints info
        if global_step % self.FLAGS.print_every == 0:
          logging.info(
            f"epoch {epoch}, "
            f"global_step {global_step}, "
            f"loss {loss}, "
            f"exp_loss {exp_loss}, "
            f"grad norm {grad_norm}, "
            f"param norm {param_norm}")

        # Sometimes saves model
        if (global_step % self.FLAGS.save_every == 0
            or global_step == msbg.get_num_batches()):
          self.saver.save(sess, checkpoint_path, global_step=global_step)

        # Sometimes evaluates model on dev loss, train F1/EM and dev F1/EM
        if global_step % self.FLAGS.eval_every == 0:
          # Logs loss for entire dev set to TensorBoard
          dev_loss = self.calculate_loss(sess,
                                         dev_input_paths,
                                         dev_target_mask_paths,
                                         "dev",
                                         self.FLAGS.dev_num_samples)
          logging.info(f"epoch {epoch}, "
                       f"global_step {global_step}, "
                       f"dev_loss {dev_loss}")
          utils.write_summary(dev_loss,
                              "dev/loss",
                              summary_writer,
                              global_step)

          # Logs dice coefficient on train set to TensorBoard
          train_dice = self.calculate_dice_coefficient(sess,
                                                       train_input_paths,
                                                       train_target_mask_paths,
                                                       "train")
          logging.info(f"epoch {epoch}, "
                       f"global_step {global_step}, "
                       f"train dice_coefficient: {train_dice}")
          utils.write_summary(train_dice,
                              "train/dice",
                              summary_writer,
                              global_step)

          # Logs dice coefficient on dev set to TensorBoard
          dev_dice = self.calculate_dice_coefficient(sess,
                                                     dev_input_paths,
                                                     dev_target_mask_paths,
                                                     "dev")
          logging.info(f"epoch {epoch}, "
                       f"global_step {global_step}, "
                       f"dev dice_coefficient: {dev_dice}")
          utils.write_summary(dev_dice,
                              "dev/dice",
                              summary_writer,
                              global_step)
      # end for batch in sbg.get_batch
    # end while num_epochs == 0 or epoch < num_epochs
    sys.stdout.flush()