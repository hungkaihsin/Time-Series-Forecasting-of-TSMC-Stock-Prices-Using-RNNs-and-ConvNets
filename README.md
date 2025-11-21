# Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets

**Live Demo Here:** [https://time-series-backend.web.app/](https://time-series-backend.web.app/)

This project predicts TSMC's next-day closing stock price using historical financial data from Yahoo Finance. It evaluates multiple deep learning models—including LSTM, GRU, Conv1D, and Feedforward Neural Networks (FFN)—to determine the most effective approach for forecasting. The final solution is deployed via a Flask API and features a modern React-based frontend for interaction and visualization.

## About The Project

This project demonstrates expertise in time series forecasting, deep learning, and full-stack deployment. It covers data preprocessing, model training, evaluation, and serving predictions through a REST API. The backend is powered by Flask, while the frontend is built with React and Vite.

## Key Features

- **End-to-End Deployment:** Deployed the Flask backend to Google Cloud Run and the React frontend to Firebase Hosting for a scalable and robust live application.
- **Containerization with Docker:** Implemented Docker for both frontend and backend, ensuring consistent development, testing, and deployment environments.
- **Continuous Integration/Continuous Deployment (CI/CD):** Established GitHub Actions workflows for automated Docker image builds and testing on every code push, enhancing code quality and reliability.
- **Backtesting Engine:** Developed a simulation engine to evaluate trading strategies based on model predictions against a buy-and-hold approach, providing quantifiable performance metrics.
- **Data Preprocessing:** Cleans and formats Yahoo Finance historical stock data for model input.  
- **Sequence Generation:** Converts time series into supervised learning format.  
- **Model Comparison:** Evaluates LSTM, GRU, Conv1D, and FFN architectures.  
- **Hyperparameter Tuning:** Optimizes model configurations for improved performance.  
- **Prediction API:** Flask-based REST endpoints to serve predictions.  
- **Interactive Frontend:** Responsive React UI for visualization and interaction with forecasts.  

## Resume-Ready Accomplishments

*   **Built a backtesting engine** to simulate trading strategies for time series forecasting models, demonstrating a potential 14.25% ROI with the FFN model (though outperformed by a buy-and-hold strategy in this specific simulation).
*   **Integrated Docker for containerization** and established a CI/CD pipeline using GitHub Actions to automate builds and testing, enhancing deployment reliability and development workflow.
*   **Architected and deployed a full-stack application** with a React frontend on Firebase Hosting and a Flask backend on Google Cloud Run, showcasing proficiency in cloud-native development and scalable infrastructure.

## Tech Stack

- **Frontend:** React, Vite, CSS (Deployed on Firebase Hosting)
- **Backend:** Python, Flask, TensorFlow/Keras (Deployed on Google Cloud Run)
- **AI/ML:** TensorFlow/Keras (LSTM, GRU, Conv1D, FFN)  
- **Data Source:** Yahoo Finance API  
- **Deployment & DevOps:** Docker, Docker Compose, Google Cloud Run, Firebase Hosting, GitHub Actions

## Getting Started

To get a local copy up and running quickly with Docker, follow these simple steps.

### Prerequisites

- Docker and Docker Compose installed.
- Node.js and npm (if you need to develop the frontend locally without Docker).
- Python 3.x and pip (if you need to develop the backend locally without Docker).

### Local Installation & Run with Docker

1. **Clone the repo**
    ```sh
    git clone https://github.com/your-username/tsmc_stock_forecasting.git
    cd tsmc_stock_forecasting
    ```

2. **Build and Start with Docker Compose**
    *   Navigate to the project's root directory (where `docker-compose.yml` is located).
    *   Build the images (this might take a few minutes the first time):
        ```sh
        docker-compose build
        ```
    *   Start both the backend and frontend services:
        ```sh
        docker-compose up
        ```
    *   Access the application in your browser at `http://localhost:8080`.

3.  **Stop the services:**
    *   To stop and remove the containers, press `Ctrl+C` in the terminal where `docker-compose up` is running, then run:
        ```sh
        docker-compose down
        ```

## Model Performance Summary

| **Model** | **Val MAE ($ / %)** | **Test MAE ($ / %)** | **Comment**                          |
|-----------|---------------------|----------------------|---------------------------------------|
| **LSTM**  | 1.40 (0.73%)         | 6.56 (3.40%)          | Overfits despite strong training     |
| **GRU**   | 1.40 (0.72%)         | 4.27 (2.21%)          | Balanced fit and generalization      |
| **Conv1D**| 2.81 (1.46%)         | 8.81 (4.56%)          | Underperforms on both sets           |
| **FFN**   | 1.66 (0.86%)         | 4.24 (2.19%)          | Best generalization performance      |

## Conclusion

The **GRU** and **FFN** models demonstrate the most reliable generalization to unseen data and are recommended for deployment. While LSTM showed strong validation performance, it exhibited signs of overfitting.

## Contact

Daniel – [k_hung2@u.pacific.edu](mailto:k_hung2@u.pacific.edu)  

Project Link: [https://github.com/hungkaihsin/tsmc_stock_forecasting](https://github.com/hungkaihsin/tsmc_stock_forecasting)