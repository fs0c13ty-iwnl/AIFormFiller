# AIFormFiller

This is my honours thesis project. The program using NLP and LLM techniques automatically fill the PDF forms according to given information from other filled forms.

## Deployment Guide

### Step 1: Download the Pre-trained Dictionary

The program uses `glove.6B.300d` as a dictionary for word embedding. Please use the provided link in `glove.6B.300d_DOWNLOAD_LINK.txt` to download the pre-trained dictionary and place it in the project's root directory.

### Step 2: Set Up OpenAI API Key

Users need to input their OpenAI API key into the environment variables.

- For Unix/Linux/Mac users, you can use the command:
  ```
  export OPENAI_API_KEY="YOUR OPENAI API KEY"
  ```

- For Windows users, you can use:
  ```
  set OPENAI_API_KEY="YOUR OPENAI API KEY"
  ```

## Input and Output Forms

The project provides three input forms and one blank output form. After the program finishes running, it will automatically generate an `Output_form_result.pdf` to display the filled form results.

The input forms are:
- `example_form1.pdf`
- `example_form2.pdf`
- `example_form3.pdf`

The blank output form is:
- `Output_form.pdf`

## Usage

1. Ensure that you have downloaded the pre-trained dictionary and placed it in the project's root directory.
2. Set up your OpenAI API key as an environment variable.
3. Run the `main.py` script to start the program.
4. The program will process the input forms and generate the filled output form as `Output_form_result.pdf`.

## Dependencies

The project relies on the following dependencies:
- Python 3.x
- OpenAI Python library
- NumPy
- pdfrw
- pdfminer

You can install the required dependencies using pip:
```
pip install openai numpy pdfrw pdfminer.six
```

## License

This project is licensed under the [MIT License](LICENSE).
