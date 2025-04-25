from process_data.utils import load_files, get_train_test

load_files()

for fold in range(5):
    train_image_paths, val_image_paths, train_labels, val_labels, train_CaseIds, val_CaseIds = get_train_test(fold)
    print(len(train_image_paths), len(val_image_paths), len(train_labels), len(val_labels))
    print(set(train_CaseIds))
    print(set(val_CaseIds))
    if any(item in set(val_CaseIds) for item in set(train_CaseIds)):
        print("There is overlap.")
    else:
        print("No overlap.")

