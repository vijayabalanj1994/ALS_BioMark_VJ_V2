from process_data.utils import load_files, get_train_test
from process_data.augemt_data import augment_images

print("\nLoadind data...")
load_files()

print("\nSplitting to train and test data...")
train_image_paths, test_image_paths, train_labels, test_labels = get_train_test(0)

print("\nNext...")
train_image_paths, train_labels = augment_images(train_image_paths, train_labels)


