import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def load_images_from_folder(folder, convert_to_gray=True, resize_dim=(64, 64)):
    image_data = []
    image_files = sorted(os.listdir(folder))
    for image_name in image_files:
        img = Image.open(os.path.join(folder, image_name))
        if convert_to_gray:
            img = img.convert('L')
        img = img.resize(resize_dim)
        img_arr = np.array(img).flatten()
        image_data.append(img_arr)
    return np.array(image_data), image_files

def perform_pca(data_matrix):
    mean_centered_data = data_matrix - np.mean(data_matrix, axis=0)
    covariance_matrix = np.cov(mean_centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return mean_centered_data, covariance_matrix, eigenvalues, eigenvectors

def select_pca_basis(eigenvalues, eigenvectors, energy_threshold=0.99):
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    total_variance = np.sum(sorted_eigenvalues)
    cumulative_variance = np.cumsum(sorted_eigenvalues) / total_variance
    num_components = np.argmax(cumulative_variance >= energy_threshold) + 1

    pca_basis = sorted_eigenvectors[:, :num_components]
    return pca_basis

def project_and_backproject(test_img, pca_basis):
    projected = np.dot(test_img, pca_basis)
    back_projected = np.dot(projected, pca_basis.T)
    return back_projected

def find_loss(original_img, back_projected_img):
    return np.sum((original_img - back_projected_img)**2)

def identify_and_compare(train_folder, test_folder):
    train_data, train_labels = load_images_from_folder(train_folder)
    _, _, eigenvalues, eigenvectors = perform_pca(train_data)
    pca_basis = select_pca_basis(eigenvalues, eigenvectors)

    test_data, test_labels = load_images_from_folder(test_folder)

    overall_min_loss = np.inf
    overall_min_label = None
    overall_min_test_label = None

    for i, test_img in enumerate(test_data):
        back_projected_img = project_and_backproject(test_img, pca_basis)
        loss = find_loss(test_img, back_projected_img)

        min_loss = np.inf
        min_label = None
        for j, train_img in enumerate(train_data):
            train_back_projected = project_and_backproject(train_img, pca_basis)
            train_loss = find_loss(test_img, train_back_projected)
            if train_loss < min_loss:
                min_loss = train_loss
                min_label = train_labels[j]

        print(f"Test Image: {test_labels[i]}, Loss: {loss}, Most Similar Train Image: {min_label}")

        if min_loss < overall_min_loss:
            overall_min_loss = min_loss
            overall_min_label = min_label
            overall_min_test_label = test_labels[i]

    print(f"\nOverall Most Similar Train Image to all Test Images: {overall_min_label} (Test Image: {overall_min_test_label})")
    display_image(os.path.join(train_folder, overall_min_label))

def display_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def main():
    print("Select an operation:")
    print("1) Face Identification")
    print("2) Detect Glasses")
    print("3) Detect Shade (there must be a shaded image in test folder)")
    print("4) Emotions")
    operation = int(input("Enter the serial number from the list: "))

    if operation == 1:
       directories = [f"images/person_{i}" for i in range(1, 5)]
    elif operation == 2:
        directories = ["images/noglasses", "images/glasses"]
    elif operation == 3:
        directories = ["images/left", "images/right"]
    elif operation == 4:
        directories = ["images/happy", "images/sad", "images/winkle", "images/sleepy", "images/normal", "images/surprise"]
    else:
        raise ValueError("Invalid operation selected.")

    pca_bases = {}
    for dir in directories:
        images, _ = load_images_from_folder(dir)  # Assuming this function returns (image_data, image_files)
        mean_centered_data, _, eigenvalues, eigenvectors = perform_pca(images)
        selected_basis = select_pca_basis(eigenvalues, eigenvectors)
        pca_bases[dir.split('/')[-1]] = (selected_basis, np.mean(images, axis=0))

    test_images, test_labels = load_images_from_folder("test")

    for idx, (test_image, label) in enumerate(zip(test_images, test_labels)):
        best_match = None
        min_loss = float('inf')
        for pca_label, (pca_basis, mean_image) in pca_bases.items():
            train_back_projected = project_and_backproject(test_image - mean_image, pca_basis)
            loss = find_loss(test_image, train_back_projected + mean_image)
            if loss < min_loss:
                min_loss = loss
                best_match = pca_label

        # Displaying each test image with caption
        plt.imshow(test_image.reshape(64,64), cmap='gray')  
        plt.title(f"Test Image {label}: Best Match - {best_match}, Loss - {min_loss}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()