class config():
    
    data_file = 'data/data_cf_all/'
    covariates_file = data_file+'x.csv'
    number_of_covariates = 58
    number_of_covariates_transformed = 78
    number_of_units = 4802
    feature_cols = ['x_'+str(i) for i in range(1, number_of_covariates+1)]
                
    outcome_file_dir = ['data/data_cf_all/'+str(i)+'/' for i in range(1, 78)]
    
    train_epochs = 1000
    train_batches = 100
    
    lambda_1 = 1
    
    autoencoder_model_dir = 'models_final/autoencoder/'
    propensity_model_dir = 'models_final/propensity/'
    counterfactual_model_dir = 'models_final/counterfactual_models/'

