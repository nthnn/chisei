<br/>
<p align="center">
    <img src="assets/chisei.png" />
    <br/>
</p>

<h1 align="center">Lightweight AI/ML Framework</h1>

Chisei (智成, meaning "Intelligence Accomplished" in Japanese) is a lightweight, efficient, and user-friendly C++ library for creating, training, and deploying fully connected neural networks. Designed with simplicity and performance in mind, Chisei is ideal for researchers, developers, and enthusiasts who want to integrate neural network functionality into their C++ applications without the overhead of larger machine learning frameworks.

## 🌟 Features

- **Feedforward Neural Networks**: Build fully connected neural networks with customizable architectures.
- **Custom Activation Functions**: Use any activation function and its derivative, allowing for flexibility and experimentation.
- **Training with Backpropagation**: Train networks using mean squared error (MSE) and gradient descent optimization.
- **Model Persistence**: Save and load models easily for reuse and deployment.
- **Lightweight Design**: Minimal external dependencies, making Chisei easy to integrate into existing C++ projects.
- **CPU Optimizations**: Optimized for CPU performance, with potential for GPU extensions.

## 🚀 Getting Started

1. Download the `*.deb` file for your system architecture from [release](https://github.com/nthnn/chisei/releases).
2. Install the `*.deb` file using the `dpkg` command on the terminal:

    ```bash
    sudo dpkg -i chisei_*.deb
    ```

3. Try to compile the examples within this repository using the `g++` command.

    ```bash
    g++ -o dist/basic_example examples/basic_example.cpp -lchisei
    g++ -o dist/mnist_example examples/mnist_example.cpp -lchisei
    ```

4. Check the **chisei** documentations at [https://chisei.vercel.app](https://chisei.vercel.app).

## 📜 License

Chisei is licensed under the [BSD 2-Clause "Simplified" License](LICENSE). You are free to use, modify, and distribute the library.
