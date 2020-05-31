import matplotlib.pyplot as plt
import classifier_mnist_activelearning
import classifier_mnist

if __name__ == '__main__':
    least_confidence_sampling_data = classifier_mnist_activelearning.main("1")
    plt.plot(least_confidence_sampling_data[0], least_confidence_sampling_data[1], "r-", label="least_confidence")
    margin_sampling_data = classifier_mnist_activelearning.main("2")
    plt.plot(margin_sampling_data[0], margin_sampling_data[1], "g-", label="margin_sampling")
    entropy_sampling_data = classifier_mnist_activelearning.main("3")
    plt.plot(entropy_sampling_data[0], entropy_sampling_data[1], "b-", label="entropy_sampling")
    num_of_data, random_sampling_average_precision = classifier_mnist.main()
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.xlabel("Number of samples trained on")
    plt.ylabel("Accuracy")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    plt.legend()
    plt.show()
