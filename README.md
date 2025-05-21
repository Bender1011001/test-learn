# CAMEL Extensions GUI

This project provides a Streamlit-based Graphical User Interface (GUI) for managing, observing, and improving CAMEL AI agents. It focuses on facilitating workflows like the proposer-executor loop and streamlining the Direct Preference Optimization (DPO) training process.

## Features (MVP)

*   **Workflow Execution:** Initiate and observe pre-defined CAMEL agent workflows in real-time.
*   **Configuration Management:** View and modify configurations for agents and workflows (e.g., LLM models, adapters) via a user-friendly interface.
*   **Log Exploration & Annotation:** Review historical agent interactions and create preference data for DPO training.
*   **DPO Training Initiation:** Configure and start DPO training runs for agents using annotated data.

## Setup Instructions

### Prerequisites

* Python 3.10-3.12 (camel-ai is not compatible with Python 3.13+)
* pip (for installing dependencies)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Bender1011001/test-learn.git
    cd test-learn 
    ```
    (Note: If your local project directory is still named `camel`, navigate into that instead of `test-learn` after cloning).

2.  **Create and activate a Python virtual environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** The camel-ai library requires Python 3.10-3.12 and is not compatible with Python 3.13+. If you encounter installation errors, please ensure you're using a compatible Python version.

4.  **API Keys (Optional but Recommended):**
    Set up necessary API keys as environment variables (e.g., `OPENAI_API_KEY`). The specific keys required will depend on the Large Language Models (LLMs) you configure for your agents in `configs/agents.yaml`.

## Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run gui/app.py
```

This will typically open the GUI in your default web browser.

## Project Structure

*   `gui/`: Contains the Streamlit application code.
    *   `app.py`: Main application entry point and session state management.
    *   `views/`: Individual page views (Dashboard, Configuration, Log Explorer, DPO Training, Settings).
*   `configs/`: Holds configuration files.
    *   `agents.yaml`: Defines agent settings, workflow sequences, and LLM configurations.
*   `scripts/`: Contains utility and operational scripts.
    *   `train_dpo.py`: Script for DPO fine-tuning (to be adapted/used by the GUI).
*   `models/`: Default directory where trained DPO adapters will be saved by the training script.
*   `logs/`: Default directory for storing application logs, including interaction logs (e.g., `camel_logs.db`) and annotations (e.g., `annotations.db`).
*   `requirements.txt`: Lists Python dependencies for the project.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: This file.

## Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` (to be created) for guidelines.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file in the root directory for more details.