# coding: utf-8

import tensorflow as tf
import sonnet as snt
import csv


class ImageSupplier(snt.AbstractModule):
    def __init__(self, image_path_file, name='ImageSupplier', image_size=64, is_header=True, batch_size=100):
        self._image_size = image_size
        self._batch_size = batch_size
        super(ImageSupplier, self).__init__(name=name)

        with open(image_path_file, "r") as f:
            reader = csv.reader(f)
            if is_header:
                _ = next(reader)
            paths = [row[0] for row in reader]

        self._image_path = tf.constant(paths)
        self._num_examples = len(paths)

        self.dataset = tf.data.Dataset.from_tensor_slices(self._image_path).map(self._parse)
        self.dataset = self.dataset.shuffle(self._num_examples).repeat().batch(self._batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()

    def _parse(self, filename):
        image_str = tf.read_file(filename)
        image = tf.image.decode_image(image_str, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, target_height=self._image_size, target_width=self._image_size)

        return image

    def _resize_image(self, image):
        h, w, c = image.shape
        crop_size = tf.minimum(h, w)
        cropped = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        resize_img = tf.image.resize_images(cropped, size=(self._image_size, self._image_size))
        return resize_img

    def _build(self):
        next_element = self.iterator.get_next()
        normalized = tf.cast(next_element, tf.float32) / 255.
        return normalized


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    supplier = ImageSupplier(image_path_file='image_path.csv', image_size=128, batch_size=3, is_header=True)
    x = supplier()
    tf.summary.image(name='input_image', tensor=x)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('test_image', sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        img = sess.run(x)
        print(img.shape)
        summary = sess.run(merged)
        writer.add_summary(summary, 0)
