class MetaSliceBatchGenerator(SliceBatchGenerator):
  def __init__(self,
               input_path_lists,
               target_mask_path_lists,
               batch_size,
               max_num_refill_batches=1000,
               num_samples=None,
               shape=(197, 233),
               shuffle=False,
               use_fake_target_masks=False):
    super().__init__(input_path_lists,
                     target_mask_path_lists,
                     batch_size,
                     max_num_refill_batches,
                     num_samples,
                     shape,
                     shuffle,
                     use_fake_target_masks)
    self.meta = {}
    with open('../data/ATLAS_R1.1/ATLAS_Meta-Data_Release_1.1_standard_mni.csv', mode='r') as infile:
      rows = [row for row in csv.reader(infile)]
      for row in rows[1:]:
        key = '/'.join([row[0], '0'+row[1], row[2].strip()])
        self.meta[key] = tuple(map(int, row[3:7]))

  def refill_batches(self):
    """
    Refills {self._batches}.
    """
    if self._pointer >= len(self._input_path_lists):
      return

    examples = []  # A Python list of (input, target_mask) tuples

    # {start_idx} and {end_idx} are values like 2000 and 3000
    # If shuffle=True, then {self._order} is a list like [56, 720, 12, ...]
    # {path_indices} is the sublist of {self._order} that represents the
    #   current batch; in other words, the current batch of inputs will be:
    #   [self._input_path_lists[path_indices[0]],
    #    self._input_path_lists[path_indices[1]],
    #    self._input_path_lists[path_indices[2]],
    #    ...]
    # {input_path_lists} and {target_mask_path_lists} are lists of paths corresponding
    #   to the indices given by {path_indices}
    start_idx, end_idx = self._pointer, self._pointer + self._max_num_refill_batches
    path_indices = self._order[start_idx:end_idx]
    input_path_lists = [
      self._input_path_lists[path_idx] for path_idx in path_indices]
    target_mask_path_lists = [
      self._target_mask_path_lists[path_idx] for path_idx in path_indices]
    zipped_path_lists = zip(input_path_lists, target_mask_path_lists)

    # Updates self._pointer for the next call to {self.refill_batches}
    self._pointer += self._max_num_refill_batches

    for input_path_list, target_mask_path_list in zipped_path_lists:
      if self._use_fake_target_masks:
        input = Image.open(input_path_list[0]).convert("L")
        # Image.resize expects (width, height) order
        examples.append((
          # np.asarray(input.resize(self._shape[::-1], Image.NEAREST)),
          np.asarray(input.crop((0, 0) + self._shape[::-1])),
          np.zeros(self._shape),
          input_path_list[0],
          "fake_target_mask"
        ))
      else:
        # Assumes {input_path_list} is a list with length 1;
        # opens input, resizes it, converts to a numpy array
        input = Image.open(input_path_list[0]).convert("L")
        # input = input.resize(self._shape[::-1], Image.NEAREST)
        input = input.crop((0, 0) + self._shape[::-1])
        input = np.asarray(input) / 255.0
        key = input_path_list[0][19:35]
        input_meta = np.array(self.meta[key])
        input_meta = np.repeat(input_meta[np.newaxis,:], self._shape[1], axis=0)
        input_meta = np.repeat(input_meta[np.newaxis,:], self._shape[0], axis=0)
        input = np.append(input[:, :, np.newaxis], input_meta, axis=2)

        # Assumes {target_mask_path_list} is a list of lists, where the outer
        # list has length 1 and the inner list has length >= 1;
        # Merges target masks if list contains more than one path
        target_mask_list = list(map(
          lambda target_mask_path: Image.open(target_mask_path).convert("L"),
          target_mask_path_list[0]))
        target_mask_list = list(map(
          # lambda target_mask: target_mask.resize(self._shape[::-1], Image.NEAREST),
          lambda target_mask: target_mask.crop((0, 0) + self._shape[::-1]),
          target_mask_list))

        target_mask_list = list(map(
          # lambda target_mask: (np.asarray(target_mask.resize(self._shape[::-1], Image.NEAREST)) > 1e-8) + 0.0,
          lambda target_mask: (np.asarray(target_mask)) / 255.0,
          target_mask_list))
        target_mask = np.minimum(np.sum(target_mask_list, axis=0), 1.0)

        # Image.resize expects (width, height) order
        examples.append((
          input,
          # Converts all values >0 to 1s
          target_mask,
          input_path_list[0],
          target_mask_path_list[0]
        ))
      if len(examples) >= self._batch_size * self._max_num_refill_batches:
        break

    for batch_start_idx in range(0, len(examples), self._batch_size):
      (inputs_batch, target_masks_batch, input_paths_batch,
       target_mask_path_lists_batch) =\
         zip(*examples[batch_start_idx:batch_start_idx+self._batch_size])
      self._batches.append((np.asarray(inputs_batch),
                            np.asarray(target_masks_batch),
                            input_paths_batch,
                            target_mask_path_lists_batch))

