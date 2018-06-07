class MetaUNet(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      # Conv
      conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 5, 64], scope_name="conv1")  # (b, 232, 196, 64)
      drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
      conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, 64, 64], scope_name="conv2")  # (b, 232, 196, 64)
      drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

      pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, 116, 98, 64)
      conv3 = self.conv2d_relu(pool1, filter_shape=[3, 3, 64, 128], scope_name="conv3")  # (b, 116, 98, 128)
      drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
      conv4 = self.conv2d_relu(drop3, filter_shape=[3, 3, 128, 128], scope_name="conv4")  # (b, 116, 98, 128)
      drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

      pool2 = self.maxpool2d(conv4, scope_name="pool2")  # (b, 58, 49, 128)
      conv5 = self.conv2d_relu(pool2, filter_shape=[3, 3, 128, 256], scope_name="conv5")  # (b, 58, 49, 256)
      drop5 = self.dropout(conv5, keep_prob=self.keep_prob, scope_name="drop5")
      conv6 = self.conv2d_relu(drop5, filter_shape=[3, 3, 256, 256], scope_name="conv6")  # (b, 58, 49, 256)
      drop6 = self.dropout(conv6, keep_prob=self.keep_prob, scope_name="drop6")

      # Deconv
      up1 = self.upsample(drop6, scope_name="up1", factor=[2, 2])  # (b, 116, 98, 256)
      deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=128, scope_name="deconv1")  # (b, 116, 98, 128)
      concat1 = tf.concat([drop4, deconv1], axis=3)  # (b, 116, 98, 256)
      conv7 = self.conv2d_relu(concat1, filter_shape=[3, 3, 256, 128], scope_name="conv7")  # (b, 116, 98, 128)
      drop7 = self.dropout(conv7, keep_prob=self.keep_prob, scope_name="drop7")
      conv8 = self.conv2d_relu(drop7, filter_shape=[3, 3, 128, 128], scope_name="conv8")  # (b, 116, 98, 128)
      drop8 = self.dropout(conv8, keep_prob=self.keep_prob, scope_name="drop8")

      up2 = self.upsample(drop8, scope_name="up2", factor=[2, 2])  # (b, 232, 196, 128)
      deconv2 = self.deconv2d(up2, filter_shape=[2, 2], num_outputs=64, scope_name="deconv2")  # (b, 232, 196, 64)
      concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, 232, 196, 128)
      conv9 = self.conv2d_relu(concat2, filter_shape=[3, 3, 128, 64], scope_name="conv9")  # (b, 232, 196, 64)
      drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
      conv10 = self.conv2d_relu(drop9, filter_shape=[3, 3, 64, 64], scope_name="conv10")  # (b, 232, 196, 64)
      drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

      conv11 = self.conv2d(drop10, filter_shape=[1, 1, 64, 1], scope_name="conv11")  # (b, 232, 196, 1)
      out = tf.identity(conv11, name="out")
      
    return out
