# Diverse_Models_for_Streamflow_Prediction
This project provides three solutions for basin runoff prediction modeling, including process-based models, LSTM models, and hybrid models.
## Project Structure
### Core Scripts

1.process-based model.py  
  Purpose: Training a process-based hydrological model for runoff prediction.  
  Key Functionality: Can choose to use the EXP-HYDRO model or the Xin’anjiang model. 
  
2.pure_LSTM.py  
  Purpose: Use the LSTM model to predict runoff, with input including the meteorological forcing data and static attribute data of the basin. 
  Key Functionality: Read data, train model, save model.  
  
3.dPL.py  
  Purpose: Training differentiable hybrid models that couples LSTM and process-based models.  
  Key Functionality: Read data, train model, save model.  
  
4.IN_LSTM.py  
  Purpose: Training alternative hybrid models that couples LSTM and process-based models.
  Key Functionality: Read data, train model, save model.
  
5.dPL_class.py  
  Purpose: Define the EXP-HYDRO model and Xin'anjiang model for hybrid modeling.  
  Key Functionality: Ensures data normalization, feature extraction, and input formatting. 
  
6.cn_class.py  
  Purpose: Define the EXP-HYDRO model and Xin'anjiang model for process-based modeling.  
  Key Functionality: Ensures data normalization, feature extraction, and input formatting. 
  
7.loss.py  
  Purpose: Implements custom loss functions for model optimization.  
  Key Functionality: Supports physics-informed loss terms to enhance prediction accuracy.  

## Installation and Setup
### Prerequisites
  Python: Version 3.8 or later.    
  Recommended packages: numpy, pandas, torch, matplotlib, and scipy.    
### Setup Steps
  1.Clone the repository:    
    git clone https://github.com/Yq-H47/Diverse_Models_for_Streamflow_Prediction
    cd Diverse_Models_for_Streamflow_Prediction  
  2.Install dependencies:  
    pip install -r environments.yml  
    (If environments.yml is not provided, manually install the required libraries.)  
  3. Prepare datasets:  
    Place your meteorological and hydrological data files in the data/ directory.  
    Ensure data format matches the specifications in dataprocess.py. 
## Data
https://github.com/Yq-H47/Catchment-Attributes-and-Meteorology-dataset-for-China-544-basins

## Outputs  
  **Metrics**: Evaluation metrics such as Nash-Sutcliffe Efficiency (NSE).  
  **Visualizations**: Predicted versus observed hydrological outputs.  
  **Logs**: Detailed logs of training and testing performance.  
## Notes and Limitations  
    Ensure datasets are preprocessed correctly using the provided scripts.  
    Modify hyperparameters in training scripts for optimal performance on your dataset.  
    For large datasets, training may require substantial computational resources.  
## Contact
  For further assistance or implementation details, please contact the author at: lh_mygis@163.com or 17563403791@163.com  
