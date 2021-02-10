"""yelp_dataset dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import os
from glob import glob

# TODO(yelp_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(yelp_dataset): BibTeX citation
_CITATION = """
"""


class YelpDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for yelp_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self):
    """Returns the dataset metadata."""
    # TODO(yelp_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(names=['food', 'drink', 'menu', 'inside', 'outside']),
        }),
        supervised_keys=("image", "label"),
        homepage="https://www.yelp.com/dataset/download"
    )

  def _split_generators(self, dl_manager):
    """Download the data and define splits."""
    # extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    json_file = '/home/bashir/Desktop/yelp/data/photos.json'
    self.df = pd.read_json(json_file, lines=True)
    self.classnames = {"food": 0, "drink": 1, "menu": 2, "inside": 3, "outside": 4}
    return {
        'train': self._generate_examples(path='/home/bashir/Desktop/yelp/data/photos'),
    }

  def _generate_examples(self, path):
    """Generator of examples for each split."""
    for img_path in glob(os.path.join(path, '*.jpg')):
      path_dir, filename = os.path.split(img_path)
      filename, ext = os.path.splitext(filename)
      label = self.classnames[self.df[self.df['photo_id'] == filename]['label'].item()]

      yield img_path, {
          'image': img_path,
          'label': label,
      }