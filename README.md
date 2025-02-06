```md
# Persian AI Model Manager Documentation

Welcome to the official documentation for the **Persian AI Model Manager**. This guide explains in detail the algorithms, workflow, and commands used to create and use AI models with Persian AI. The system offers both a browser-based interface and a terminal-based interface, providing flexibility and control over model training and deployment.

---

## Table of Contents

- [Overview](#overview)
- [Creating Models](#creating-models)
  - [Using the Browser Interface (`learnweb`)](#using-the-browser-interface-learnweb)
  - [Using the Terminal Interface (`learnchat`)](#using-the-terminal-interface-learnchat)
- [Using Models](#using-models)
  - [Listing Models](#listing-models)
  - [Running Model in Browser (`runweb`)](#running-model-in-browser-runweb)
  - [Running Model in Terminal (`runchat`)](#running-model-in-terminal-runchat)
- [Algorithmic Workflow](#algorithmic-workflow)
- [Conclusion](#conclusion)

---

## Overview

The Persian AI Model Manager provides a powerful, yet user-friendly environment for:
- **Creating New Models:** Train models interactively via a browser or directly from the terminal.
- **Using Models:** Deploy and interact with your trained models either in a browser-based chat interface or directly via the terminal.

This documentation covers all commands and workflows required to fully leverage the system.

---

## Creating Models

### Using the Browser Interface (`persianai learnweb`)

- **Command:**  
  ```bash
  persianai learnweb
  ```
- **Description:**  
  Launches an interactive web-based interface that allows you to create a new AI model directly in your browser. This mode is ideal for users who prefer a graphical user interface for setting up and monitoring the training process.

- **Workflow:**
  1. **User Initiation:** Execute `persianai learnweb` from the terminal.
  2. **Browser Launch:** The system automatically opens a browser window with the model creation interface.
  3. **Data Input:** Fill in the form with model details (e.g., model name, parameters, dataset selection).
  4. **Training Process:** The interface displays real-time progress of the training algorithm.
  5. **Model Registration:** Once training is complete, the new model is saved and added to your model list.

---

### Using the Terminal Interface (`persianai learnchat`)

- **Basic Command:**  
  ```bash
  persianai learnchat MODELNAME INFORMATION_FOLDER_PATH
  ```
- **Advanced Command (for more control):**  
  ```bash
  persianai learnchat MODELNAME INFORMATION_FOLDER_PATH accuracy_param test_size_param number_test
  ```
- **Description:**  
  Creates a new model using data from the specified folder. The basic command offers a quick way to initiate training in the terminal, while the advanced command lets you fine-tune parameters like accuracy, test size, and the number of tests for better control over the model’s performance.

- **Workflow:**
  1. **Command Execution:** Enter the appropriate command with the required parameters in the terminal.
  2. **Data Loading:** The system reads and processes data from the provided folder path.
  3. **Parameter Configuration:** Training parameters (such as desired accuracy, test size, etc.) are configured based on user input.
  4. **Model Training:** The terminal displays training progress via logs and status messages.
  5. **Model Saving:** Upon completion, the trained model is saved and registered in your model registry.

---

## Using Models

### Listing Models

- **Command:**  
  ```bash
  persianai list
  ```
- **Description:**  
  Displays a list of all the models you have created along with details such as model name, creation date, and training parameters.

---

### Running Model in Browser (`persianai runweb`)

- **Command:**  
  ```bash
  persianai runweb MODELNAME
  ```
- **Description:**  
  Launches a browser-based chat interface for the specified model. This is particularly useful for interactive testing, demonstration, and real-time interaction with your model.

- **Workflow:**
  1. **Model Retrieval:** The system retrieves the specified model.
  2. **Interface Launch:** A browser window opens with the chat interface.
  3. **Interaction:** Send and receive messages interactively via the web interface.

---

### Running Model in Terminal (`persianai runchat`)

- **Command:**  
  ```bash
  persianai runchat MODELNAME
  ```
- **Description:**  
  Runs the specified model directly in the terminal. This mode is designed for quick testing, debugging, and integration with other terminal-based workflows.

- **Workflow:**
  1. **Model Retrieval:** The system loads the model from the stored registry.
  2. **Terminal Chat:** The terminal is set up for a chat-like interaction, displaying incoming messages and accepting user input.

---

## Algorithmic Workflow

Below is a high-level overview of the algorithmic steps involved in both creating and using models:

1. **Model Creation:**
   - **Initialization:** User initiates the command (either `learnweb` or `learnchat`).
   - **Data Processing:** The system reads the dataset from a specified folder.
   - **Parameter Setup:** User-defined parameters (or defaults) are set.
   - **Training:** The model is trained using machine learning techniques. In the browser mode, this process is visualized with progress indicators; in the terminal mode, logs are printed.
   - **Registration:** Once training completes, the model is saved and registered in the model list for future use.

2. **Model Deployment and Interaction:**
   - **Model Selection:** The user lists available models using `persianai list`.
   - **Interface Launch:** The model is launched either in a web-based chat interface (`runweb`) or a terminal-based chat interface (`runchat`).
   - **Interaction Loop:** The user sends messages to the model, and the model processes the input, generates a response using its trained parameters, and returns the answer.
   - **Logging & History:** Conversation history is maintained locally (in the browser or terminal) for the duration of the session.

---

## Conclusion

The **Persian AI Model Manager** offers a comprehensive, flexible platform for creating and deploying AI models. Whether you prefer an interactive browser interface or the control of the terminal, the system provides robust tools to:
- **Create new models** with customizable parameters.
- **Deploy models** in both web and terminal environments.
- **Interact** with your models in real-time, with local history management.

For additional details, refer to our [GitHub repository](https://github.com/YourRepoLink) or contact the developer, **Mohammad Taha Gorji**.

---

*Happy Modeling!*
```
