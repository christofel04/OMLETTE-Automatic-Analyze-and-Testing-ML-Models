# Function Analyze_and_Testing_ML_Model for Automatic Analyzing and Testing ML Models Omlette
# Function Analyze_and_Testing_ML_Model for Analyze and Testing ML Models for Analyze and Testing ML Models for Classification and Analyze and Testing ML Models for Regression
# Created by : Christofel Rio Goenawan
# AI & Analytics Consultant at Hyundai and AI & Robotics Master Student at Korean Advanced Institute of Science and Technology
# If there Are A lot of Advices and Want to Discuss Automatic Analyzing and Testing ML Models Omlette Kindly Chit Chat Me on christofel.goenawan@kaist.ac.kr

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier , KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor 
from sklearn.ensemble import AdaBoostClassifier , AdaBoostRegressor
import xgboost as xgb
from sklearn.linear_model import LogisticRegression , LinearRegression


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import f1_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_log_error


def Analyze_and_Testing_ML_Model_for_Regression( ML_Dataset : pd.DataFrame  ,
                                                   ML_Column_Predicted  : str ,
                                                   ML_Key_Metrics : str = "RMSE" ,
                                                     ML_Evaluation_Cross_Validation_Number_Cross_Validation : int = 5  ) :

    # Check if ML_Column_Predicted in the ML_Dataset or Not

    if ML_Column_Predicted not in ML_Dataset.columns :

        print( "There are no ML Column Predicted in ML Dataset..../nPlease Add ML Column Predicted in ML Dataset..." )

        return
    

    # Check if the ML_Column_Predicted is Positive Value Regressior or Not Positive Value Regression

    if ML_Dataset[ ML_Dataset[ ML_Column_Predicted] < 0 ].shape[ 1 ] == ML_Dataset.shape[ 1 ] :

        is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression = True

    else :

        is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression = False

    # Check if the ML_Key_Metrics is Accuracy F1 Score or AUC Score

    if ML_Key_Metrics not in [ "RMSE" , "MAE" , "RMSLE" ]:

        print( "ML Key Matrics must be Root Mean Square Error , Mean Absolute Error or Root Mean Square Logistic Error...." )

        print( "Please Have ML_Key_Metrics 'RMSE' For Root Mean Square Error , 'MAE' For Mean Absolute Error and 'RMSLE for Root Mean Square Logistic Error" )

        return 


    if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression  == True :  

        KNN_Neighbor_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        Random_Forest_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        Adaboost_Regression_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        XGBoost_Regression_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        Linear_Regression_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

    else :

        KNN_Neighbor_ML_Perf_Accuracy = [ 0 , 0 ]

        Random_Forest_ML_Perf_Accuracy = [ 0 , 0 ]

        Adaboost_Regression_ML_Perf_Accuracy = [ 0 , 0 ]

        XGBoost_Regression_ML_Perf_Accuracy = [ 0 , 0 ]

        Linear_Regression_ML_Perf_Accuracy = [ 0 , 0 ]


    # Drop Null Values in ML_Dataset Because Machine Learning Models KNN Neighbor Machine Learning SVM and Machine Learning XGBoost cant Have Null Machine Learning Dataset Columns

    ML_Dataset = ML_Dataset.dropna()

    # Check if there is Categorical String Column in ML_Dataset

    List_of_ML_Dataset_Categorical_String_Column = ML_Dataset.select_dtypes( include=[ object, bool  ] ).columns

    if len( List_of_ML_Dataset_Categorical_String_Column ) > 0 :
        # If there is Categorical String Column in ML Dataset

        print( "---------------------------------------------" )
        print( "There is Categorical String Value in ML Dataset Column : {}".format( str( List_of_ML_Dataset_Categorical_String_Column ) ) ) 

        for ML_Dataset_Categorical_String_Column in List_of_ML_Dataset_Categorical_String_Column :

            ML_Dataset[ ML_Dataset_Categorical_String_Column ] = ML_Dataset[ ML_Dataset_Categorical_String_Column ].astype( str )

            List_of_Unique_Value_of_ML_Dataset_Categorical_String_Column = list( ML_Dataset[ ML_Dataset_Categorical_String_Column ].unique() )
            
            for ( j , Value_of_ML_Dataset_Categorical_String_Column ) in enumerate( List_of_Unique_Value_of_ML_Dataset_Categorical_String_Column  ) :
                
                ML_Dataset[ ML_Dataset_Categorical_String_Column ] = ML_Dataset[ ML_Dataset_Categorical_String_Column ].replace( { Value_of_ML_Dataset_Categorical_String_Column : str( j ) } )

            
            ML_Dataset[ ML_Dataset_Categorical_String_Column ] = ML_Dataset_Categorical_String_Column.astype( int )
                





    for i in range( ML_Evaluation_Cross_Validation_Number_Cross_Validation ) :

        print( "----------------------------------------------------------------" )
        print( "Processing ML Dataset Evaluation ML Cross Validation {}".format( i ) ) 

        X = ML_Dataset.drop([ ML_Column_Predicted ] , axis = 1 )

        y = ML_Dataset[ ML_Column_Predicted ]

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=( 1 / ML_Evaluation_Cross_Validation_Number_Cross_Validation ) )

        # Analyze and Testing ML KNN Neighbor ML Models Dataset

        neigh = KNeighborsRegressor()

        neigh.fit(X_train, y_train )

        y_pred = neigh.predict(X_test)


        KNN_Neighbor_ML_Perf_Accuracy[ 0 ] = KNN_Neighbor_ML_Perf_Accuracy[ 0 ] +  mean_squared_error( y_test , y_pred )**( 0.5 )

        KNN_Neighbor_ML_Perf_Accuracy[ 1 ] = KNN_Neighbor_ML_Perf_Accuracy[ 1 ] + mean_absolute_error( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :
            
            KNN_Neighbor_ML_Perf_Accuracy[ 2 ] = KNN_Neighbor_ML_Perf_Accuracy[ 2 ] + mean_squared_log_error( y_test , y_pred )**( 0.5 )


        # Analyze and Testing ML Random Forest Classifier ML Model

        clf = RandomForestRegressor()

        clf.fit( X_train , y_train )

        y_pred = clf.predict(X_test)

        Random_Forest_ML_Perf_Accuracy[ 0 ] = Random_Forest_ML_Perf_Accuracy[ 0 ] +  mean_squared_error( y_test , y_pred )**( 0.5 )

        Random_Forest_ML_Perf_Accuracy[ 1 ] = Random_Forest_ML_Perf_Accuracy[ 1 ] + mean_absolute_error( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :
            
            Random_Forest_ML_Perf_Accuracy[ 2 ] = Random_Forest_ML_Perf_Accuracy[ 2 ] + mean_squared_log_error( y_test , y_pred )**( 0.5 )

        # Analyze and Testing ML Adaboost Classifier ML Model

        clf = AdaBoostRegressor()

        clf.fit( X_train , y_train )

        y_pred = clf.predict(X_test)

        Adaboost_Regression_ML_Perf_Accuracy[ 0 ] = Adaboost_Regression_ML_Perf_Accuracy[ 0 ] +  mean_squared_error( y_test , y_pred )**( 0.5 )

        Adaboost_Regression_ML_Perf_Accuracy[ 1 ] = Adaboost_Regression_ML_Perf_Accuracy[ 1 ] + mean_absolute_error( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :
            
            Adaboost_Regression_ML_Perf_Accuracy[ 2 ] = Adaboost_Regression_ML_Perf_Accuracy[ 2 ] + mean_squared_log_error( y_test , y_pred )**( 0.5 )

        # Analyze and Testing ML XGBoost Classifier ML Model
        
        model = xgb.XGBRegressor()

        model.fit( X_train ,  y_train )

        y_pred = model.predict( X_test )

        XGBoost_Regression_ML_Perf_Accuracy[ 0 ] = XGBoost_Regression_ML_Perf_Accuracy[ 0 ] +  mean_squared_error( y_test , y_pred )**( 0.5 )

        XGBoost_Regression_ML_Perf_Accuracy[ 1 ] = XGBoost_Regression_ML_Perf_Accuracy[ 1 ] + mean_absolute_error( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :
            
            XGBoost_Regression_ML_Perf_Accuracy[ 2 ] = XGBoost_Regression_ML_Perf_Accuracy[ 2 ] + mean_squared_log_error( y_test , y_pred )**( 0.5 )

        # Analyze and Testing ML Linear Regression ML Model

        # Standardize the Scalar Column of ML_Dataset to Measure How Distribution of ML_Dataset

        List_of_ML_Dataset_Scalar_and_Float_Column = X_train.select_dtypes( include=[float , int] ).columns

        if len( List_of_ML_Dataset_Scalar_and_Float_Column ) > 1 :

            Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale = StandardScaler().fit( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] )

            X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ])

            X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] ) 

        
        logreg = LinearRegression()

        logreg.fit( X_train , y_train )

        y_pred = logreg.predict(X_test)

        Linear_Regression_ML_Perf_Accuracy[ 0 ] = Linear_Regression_ML_Perf_Accuracy[ 0 ] +  mean_squared_error( y_test , y_pred )**( 0.5 )

        Linear_Regression_ML_Perf_Accuracy[ 1 ] = Linear_Regression_ML_Perf_Accuracy[ 1 ] + mean_absolute_error( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :
            
            Linear_Regression_ML_Perf_Accuracy[ 2 ] = Linear_Regression_ML_Perf_Accuracy[ 2 ] + mean_squared_log_error( y_test , y_pred )**( 0.5 )


    if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :

        for ML_Evaluation_Cross_Validation in [ KNN_Neighbor_ML_Perf_Accuracy ,

    Random_Forest_ML_Perf_Accuracy ,

    Adaboost_Regression_ML_Perf_Accuracy,

    XGBoost_Regression_ML_Perf_Accuracy ,

    Linear_Regression_ML_Perf_Accuracy ] :
            
            for k in range( 3 ):

                ML_Evaluation_Cross_Validation[ k ] = ML_Evaluation_Cross_Validation[ k ]/ ML_Evaluation_Cross_Validation_Number_Cross_Validation

    else :

        for ML_Evaluation_Cross_Validation in [ KNN_Neighbor_ML_Perf_Accuracy ,

    Random_Forest_ML_Perf_Accuracy ,

    Adaboost_Regression_ML_Perf_Accuracy ,

    XGBoost_Regression_ML_Perf_Accuracy ,
    
    Linear_Regression_ML_Perf_Accuracy ] :
            
            for k in range( 2 ):

                ML_Evaluation_Cross_Validation[ k ] = ML_Evaluation_Cross_Validation[ k ]/ ML_Evaluation_Cross_Validation_Number_Cross_Validation

    # Create Analyze and Testing ML Perf ML Dataset

    if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == True :

        Analyze_and_Testing_ML_Perf_ML_Dataset = pd.DataFrame( { "KNN Neighbor Perf" : KNN_Neighbor_ML_Perf_Accuracy ,
                                                                                                    "Random Forest Perf" : Random_Forest_ML_Perf_Accuracy ,
                                                                                                    "Adaboost Perf" : Adaboost_Regression_ML_Perf_Accuracy,
                                                                                                    "XGBoost Perf" : XGBoost_Regression_ML_Perf_Accuracy,
                                                                                                    "Linear Regression Perf" : Linear_Regression_ML_Perf_Accuracy } )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.index = [ "RMSE" , "MAE" , "RMSLE"]
    
    else :

        Analyze_and_Testing_ML_Perf_ML_Dataset = pd.DataFrame( { "KNN Neighbor Perf" : KNN_Neighbor_ML_Perf_Accuracy ,
                                                                                                    "Random Forest Perf" : Random_Forest_ML_Perf_Accuracy ,
                                                                                                    "Adaboost Perf" : Adaboost_Regression_ML_Perf_Accuracy,
                                                                                                    "XGBoost Perf" : XGBoost_Regression_ML_Perf_Accuracy,
                                                                                                    "Linear Regression Perf" : Linear_Regression_ML_Perf_Accuracy } )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.index = [ "RMSE" , "MAE" ]

    
    print( Analyze_and_Testing_ML_Perf_ML_Dataset.to_string() )

    # Show Visualization Insight of Analyze and Testing ML Evaluation ML Dataset

    sns.set_theme()

    if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression == False :

        figure, axis = plt.subplots( 2 ,  1 ,figsize = ( 20 , 26 ))

        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "RMSE" ].plot( kind = "bar" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Root Mean Square Error of ML Perf ML Dataset",
                                                                                                                ax = axis[ 0  ] )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "MAE" ].plot( kind = "barh" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Mean Absolute Error of ML Perf ML Dataset",
                                                                                                                ax = axis[ 1 ] )
        

        plt.show()

    else :

        figure, axis = plt.subplots( 3 ,  1 , figsize = ( 20 , 39 ))

        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "RMSE" ].plot( kind = "bar" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Root Mean Square Error of ML Perf ML Dataset",
                                                                                                                ax = axis[ 0  ] )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "MAE" ].plot( kind = "barh" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Mean Absolute Error of ML Perf ML Dataset",
                                                                                                                ax = axis[ 1  ] )
        

        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "RMSLE" ].plot( kind = "bar" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Root Mean Square Logistic Regression Error of ML Perf ML Dataset",
                                                                                                                ax = axis[ 2 ] )
        
        plt.show()


    # Show the Insight of Analyzing and Testing ML Evaluation ML Dataset

    Analyze_and_Testing_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()

    if ML_Key_Metrics == "RMSE" :

        Best_ML_Accuracy_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset[ Analyze_and_Testing_ML_Perf_ML_Dataset[ "RMSE" ] == Analyze_and_Testing_ML_Perf_ML_Dataset[ "RMSE" ].max() ].index[ 0 ]

        Best_ML_Accuracy_ML_Perf = Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ]

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Root Mean Square Error ML Model is {} With Root Mean Square Error {} with Mean Absolute Error {} with Root Mean Square Logistic Regression Error {}".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ] ) ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSLE" ] )))
            
        else :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Root Mean Square Error ML Model is {} With Root Mean Square Error {} with Mean Absolute Error {} ".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ] ) ))
            
    elif ML_Key_Metrics == "MAE" :

        Best_ML_Accuracy_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset[ Analyze_and_Testing_ML_Perf_ML_Dataset[ "MAE" ] == Analyze_and_Testing_ML_Perf_ML_Dataset[ "MAE" ].max() ].index[ 0 ]

        Best_ML_Accuracy_ML_Perf = Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ]

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Mean Absolute Error ML Model is {} With Root Mean Square Error {} with Mean Absolute Error {} with Root Mean Square Logistic Regression Error {}".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ] ) ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSLE" ] )))
            
        else :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Mean Absolute Error Logistic Regression ML Model is {} With Root Mean Square Error {} with Mean Absolute Error {} ".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ] ) ))
            
    elif ML_Key_Metrics == "RMSLE" :

        Best_ML_Accuracy_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset[ Analyze_and_Testing_ML_Perf_ML_Dataset[ "RMSLE" ] == Analyze_and_Testing_ML_Perf_ML_Dataset[ "RMSLE" ].max() ].index[ 0 ]

        Best_ML_Accuracy_ML_Perf = Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSLE" ]

        if is_ML_Column_Predicted_ML_Dataset_Positive_Value_Regression :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Root Mean Square Logistic Error Logistic Regression ML Model is {} With Root Mean Square Error {} with Mean Absolute Error {} with Root Mean Square Logistic Error {}".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ] ) ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSLE" ] )))
            
        else :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Root Mean Square Logistic Regression Error ML Model is {} With Root Mean Square Error {} with Mean Absolute Error {} ".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "RMSE" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "MAE" ] ) ))
            
    # Train the Best ML Evaluation using ML Dataset ML Column Predicted
    
    X_train = ML_Dataset[ [ i for i in ML_Dataset.columns  if i != ML_Column_Predicted ]]
    
    #col = [ ( i + 2 )  for i in range( 5 ) if i % 2 == 0 ]

    y_train = ML_Dataset[ ML_Column_Predicted ]

    if Best_ML_Accuracy_ML_Perf_ML_Dataset == "KNN Neighbor Perf" :

        neigh = KNeighborsRegressor()

        neigh.fit( X_train, y_train )

        return(  Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    neigh  )
    

    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "Random Forest Perf" :

        clf = RandomForestRegressor()

        clf.fit( X_train , y_train )

        return(  Best_ML_Accuracy_ML_Perf_ML_Dataset , 
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    clf  )
    
    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "Adaboost Perf" :

        clf = AdaBoostRegressor()

        clf.fit( X_train , y_train )

        return(  Best_ML_Accuracy_ML_Perf_ML_Dataset , 
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    clf  )
    
    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "XGBoost Perf" :

        model = xgb.XGBRegressor()

        model.fit( X_train ,  y_train )

        return(  Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    model  )

    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "Linear Regression Perf" :

        
        List_of_ML_Dataset_Scalar_and_Float_Column = X_train.select_dtypes( include=[float , int] )

        if len( List_of_ML_Dataset_Scalar_and_Float_Column ) > 1 :

            Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale = StandardScaler().fit( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] )

            X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ])

            X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] ) 

        logreg = LinearRegression()

        logreg.fit( X_train , y_train )

        return( Best_ML_Accuracy_ML_Perf_ML_Dataset , 
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    logreg  )


    return None



def Analyze_and_Testing_ML_Model_for_Classification( ML_Dataset : pd.DataFrame , 
                                                    ML_Column_Predicted : str , 
                                                    ML_Key_Metrics = "Acc" ,
                                                    ML_Evaluation_Cross_Validation_Number_Cross_Validation = 5 ) :
    
    # Check if ML_Column_Predicted in the ML_Dataset or Not

    if ML_Column_Predicted not in ML_Dataset.columns :

        print( "There are no ML Column Predicted in ML Dataset..../nPlease Add ML Column Predicted in ML Dataset..." )

        return
    
    # Check if the ML_Column_Predicted ML_Dataset is Binary Classification or Not Binary Classification

    if len( list( ML_Dataset[ ML_Column_Predicted ].astype( int ).unique()) ) == 2 :

        is_ML_Column_Predicted_ML_Dataset_Binary_Classification = True
    
    else :

        is_ML_Column_Predicted_ML_Dataset_Binary_Classification = False 

    # Check if the ML_Key_Metrics is Accuracy F1 Score or AUC Score

    if ML_Key_Metrics not in [ "Acc" , "F1 Score" , "AUC Score" ]:

        print( "ML Key Matrics must be Accuracy , F1 Score or AUC Score...." )

        print( "Please Have ML_Key_Metrics 'Acc' For Accuracy , 'F1 Score' For F1 Score and 'AUC Score' for AUC Score " )

        return 


    if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :  

        KNN_Neighbor_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        SVM_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        Random_Forest_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        Adaboost_Classifier_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        XGBoost_Classifier_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

        Logistic_Regression_ML_Perf_Accuracy = [ 0 , 0 , 0 ]

    else :

        KNN_Neighbor_ML_Perf_Accuracy = [ 0 , 0 ]

        SVM_ML_Perf_Accuracy = [ 0 , 0 ]

        Random_Forest_ML_Perf_Accuracy = [ 0 , 0 ]

        Adaboost_Classifier_ML_Perf_Accuracy = [ 0 , 0 ]

        XGBoost_Classifier_ML_Perf_Accuracy = [ 0 , 0 ]

    # Drop Null Values in ML_Dataset Because Machine Learning Models KNN Neighbor Machine Learning SVM and Machine Learning XGBoost cant Have Null Machine Learning Dataset Columns

    ML_Dataset = ML_Dataset.dropna()

    # Check if there is Categorical String Column in ML_Dataset

    List_of_ML_Dataset_Categorical_String_Column = ML_Dataset.select_dtypes( include=[ object, bool  ] ).columns

    if len( List_of_ML_Dataset_Categorical_String_Column ) > 0 :
        # If there is Categorical String Column in ML Dataset

        print( "---------------------------------------------" )
        print( "There is Categorical String Value in ML Dataset Column : {}".format( str( List_of_ML_Dataset_Categorical_String_Column ) ) ) 

        for ML_Dataset_Categorical_String_Column in List_of_ML_Dataset_Categorical_String_Column :

            ML_Dataset[ ML_Dataset_Categorical_String_Column ] = ML_Dataset[ ML_Dataset_Categorical_String_Column ].astype( str )

            List_of_Unique_Value_of_ML_Dataset_Categorical_String_Column = list( ML_Dataset[ ML_Dataset_Categorical_String_Column ].unique() )
            
            for ( j , Value_of_ML_Dataset_Categorical_String_Column ) in enumerate( List_of_Unique_Value_of_ML_Dataset_Categorical_String_Column  ) :
                
                ML_Dataset[ ML_Dataset_Categorical_String_Column ] = ML_Dataset[ ML_Dataset_Categorical_String_Column ].replace( { Value_of_ML_Dataset_Categorical_String_Column : str( j ) } )

            
            ML_Dataset[ ML_Dataset_Categorical_String_Column ] = ML_Dataset[ ML_Dataset_Categorical_String_Column ].astype( int )
                





    for i in range( ML_Evaluation_Cross_Validation_Number_Cross_Validation ) :

        print( "----------------------------------------------------------------" )
        print( "Processing ML Dataset Evaluation ML Cross Validation {}".format( i ) ) 

        X = ML_Dataset.drop([ ML_Column_Predicted ] , axis = 1 )

        y = ML_Dataset[ ML_Column_Predicted ]

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=( 1 / ML_Evaluation_Cross_Validation_Number_Cross_Validation ) )

        # Analyze and Testing ML KNN Neighbor ML Models Dataset

        neigh = KNeighborsClassifier()

        neigh.fit(X_train, y_train )

        y_pred = neigh.predict(X_test)

        KNN_Neighbor_ML_Perf_Accuracy[ 0 ] = KNN_Neighbor_ML_Perf_Accuracy[ 0 ] +  accuracy_score( y_test , y_pred )

        KNN_Neighbor_ML_Perf_Accuracy[ 1 ] = KNN_Neighbor_ML_Perf_Accuracy[ 1 ] + f1_score( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :
        
            fpr, tpr, thresholds = metrics.roc_curve( y_test , y_pred)
            
            KNN_Neighbor_ML_Perf_Accuracy[ 2 ] = KNN_Neighbor_ML_Perf_Accuracy[ 2 ] + metrics.auc(fpr, tpr)

        # Analyze and Testing ML Supervised ML Model 

        clf = SVC()

        clf.fit( X_train , y_train )

        y_pred = clf.predict(X_test)

        SVM_ML_Perf_Accuracy[ 0 ]  = SVM_ML_Perf_Accuracy[ 0 ] +  accuracy_score( y_test , y_pred )

        SVM_ML_Perf_Accuracy[ 1 ] = SVM_ML_Perf_Accuracy[ 1 ] + f1_score( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :
        
            fpr, tpr, thresholds = metrics.roc_curve( y_test , y_pred )
            
            SVM_ML_Perf_Accuracy[ 2 ]  = SVM_ML_Perf_Accuracy[ 2 ] + metrics.auc(fpr, tpr)

        # Analyze and Testing ML Random Forest Classifier ML Model

        clf = RandomForestClassifier()

        clf.fit( X_train , y_train )

        y_pred = clf.predict(X_test)

        Random_Forest_ML_Perf_Accuracy[ 0 ]  = Random_Forest_ML_Perf_Accuracy[ 0 ] +  accuracy_score( y_test , y_pred )

        Random_Forest_ML_Perf_Accuracy[ 1 ] = Random_Forest_ML_Perf_Accuracy[ 1 ] + f1_score( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :
        
            fpr, tpr, thresholds = metrics.roc_curve( y_test , y_pred )
            
            Random_Forest_ML_Perf_Accuracy[ 2 ]  = Random_Forest_ML_Perf_Accuracy[ 2 ] + metrics.auc(fpr, tpr)

        # Analyze and Testing ML Adaboost Classifier ML Model

        clf = AdaBoostClassifier()

        clf.fit( X_train , y_train )

        y_pred = clf.predict(X_test)

        Adaboost_Classifier_ML_Perf_Accuracy[ 0 ]  = Adaboost_Classifier_ML_Perf_Accuracy[ 0 ] +  accuracy_score( y_test , y_pred )

        Adaboost_Classifier_ML_Perf_Accuracy[ 1 ] = Adaboost_Classifier_ML_Perf_Accuracy[ 1 ] + f1_score( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :
        
            fpr, tpr, thresholds = metrics.roc_curve( y_test , y_pred )
            
            Adaboost_Classifier_ML_Perf_Accuracy[ 2 ]  = Adaboost_Classifier_ML_Perf_Accuracy[ 2 ] + metrics.auc(fpr, tpr)

        # Analyze and Testing ML XGBoost Classifier ML Model
        
        model = xgb.XGBClassifier()

        model.fit( X_train ,  y_train )

        y_pred = model.predict( X_test )

        XGBoost_Classifier_ML_Perf_Accuracy[ 0 ]  = XGBoost_Classifier_ML_Perf_Accuracy[ 0 ] +  accuracy_score( y_test , y_pred )

        XGBoost_Classifier_ML_Perf_Accuracy[ 1 ] = XGBoost_Classifier_ML_Perf_Accuracy[ 1 ] + f1_score( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :
        
            fpr, tpr, thresholds = metrics.roc_curve( y_test , y_pred )
            
            XGBoost_Classifier_ML_Perf_Accuracy[ 2 ]  = XGBoost_Classifier_ML_Perf_Accuracy[ 2 ] + metrics.auc(fpr, tpr)

        # Analyze and Testing ML Logistic Regression ML Model

        # Standardize the Scalar Column of ML_Dataset to Measure How Distribution of ML_Dataset

        List_of_ML_Dataset_Scalar_and_Float_Column = X_train.select_dtypes( include=[float , int] ).columns

        if len( List_of_ML_Dataset_Scalar_and_Float_Column ) > 1 :

            Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale = StandardScaler().fit( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] )

            X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ])

            X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] ) 

        
        logreg = LogisticRegression()

        logreg.fit( X_train , y_train )

        y_pred = logreg.predict(X_test)

        Logistic_Regression_ML_Perf_Accuracy[ 0 ]  = Logistic_Regression_ML_Perf_Accuracy[ 0 ] +  accuracy_score( y_test , y_pred )

        Logistic_Regression_ML_Perf_Accuracy[ 1 ] = Logistic_Regression_ML_Perf_Accuracy[ 1 ] + f1_score( y_test , y_pred )

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :
        
            fpr, tpr, thresholds = metrics.roc_curve( y_test , y_pred )
            
            Logistic_Regression_ML_Perf_Accuracy[ 2 ]  = Logistic_Regression_ML_Perf_Accuracy[ 2 ] + metrics.auc(fpr, tpr)
        

    if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :

        for ML_Evaluation_Cross_Validation in [ KNN_Neighbor_ML_Perf_Accuracy ,

    SVM_ML_Perf_Accuracy,

    Random_Forest_ML_Perf_Accuracy ,

    Adaboost_Classifier_ML_Perf_Accuracy ,

    XGBoost_Classifier_ML_Perf_Accuracy ,

    Logistic_Regression_ML_Perf_Accuracy ] :
            
            for k in range( 3 ):

                ML_Evaluation_Cross_Validation[ k ] = ML_Evaluation_Cross_Validation[ k ]/ ML_Evaluation_Cross_Validation_Number_Cross_Validation

    else :

        for ML_Evaluation_Cross_Validation in [ KNN_Neighbor_ML_Perf_Accuracy ,

    SVM_ML_Perf_Accuracy,

    Random_Forest_ML_Perf_Accuracy ,

    Adaboost_Classifier_ML_Perf_Accuracy ,

    XGBoost_Classifier_ML_Perf_Accuracy ] :
            
            for k in range( 2 ):

                ML_Evaluation_Cross_Validation[ k ] = ML_Evaluation_Cross_Validation[ k ]/ ML_Evaluation_Cross_Validation_Number_Cross_Validation

    # Create Analyze and Testing ML Perf ML Dataset

    if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == True :

        Analyze_and_Testing_ML_Perf_ML_Dataset = pd.DataFrame( { "KNN Neighbor Perf" : KNN_Neighbor_ML_Perf_Accuracy ,
                                                                                                    "SVM Perf" : SVM_ML_Perf_Accuracy ,
                                                                                                    "Random Forest Perf" : Random_Forest_ML_Perf_Accuracy ,
                                                                                                    "Adaboost Perf" : Adaboost_Classifier_ML_Perf_Accuracy ,
                                                                                                    "XGBoost Perf" : Adaboost_Classifier_ML_Perf_Accuracy ,
                                                                                                    "Logistic Classifier Perf" : Logistic_Regression_ML_Perf_Accuracy } )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.index = [ "Accuracy" , "F1 Score" , "AUC Score"]
    
    else :

        Analyze_and_Testing_ML_Perf_ML_Dataset = pd.DataFrame( { "KNN Neighbor Perf" : KNN_Neighbor_ML_Perf_Accuracy ,
                                                                                                    "SVM Perf" : SVM_ML_Perf_Accuracy ,
                                                                                                    "Random Forest Perf" : Random_Forest_ML_Perf_Accuracy ,
                                                                                                    "Adaboost Perf" : Adaboost_Classifier_ML_Perf_Accuracy ,
                                                                                                    "XGBoost Perf" : Adaboost_Classifier_ML_Perf_Accuracy  } )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.index = [ "Accuracy" , "F1 Score" ]

    
    print( Analyze_and_Testing_ML_Perf_ML_Dataset.to_string() )

    # Show Visualization Insight of Analyze and Testing ML Evaluation ML Dataset

    sns.set_theme()

    if is_ML_Column_Predicted_ML_Dataset_Binary_Classification == False :

        figure, axis = plt.subplots( 2 , 1 )

        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "Accuracy" ].plot( kind = "bar" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Accuracy of ML Perf ML Dataset",
                                                                                                                ax = axis[  0 ] )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "F1 Score" ].plot( kind = "barh" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "F1 Score Accuracy of ML Perf ML Dataset",
                                                                                                                ax = axis[ 1 ] )
        

        plt.show()

    else :

        figure, axis = plt.subplots( 3 ,  1 )

        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "Accuracy" ].plot( kind = "bar" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "Accuracy of ML Perf ML Dataset",
                                                                                                                ax = axis[ 0 ] )
        
        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "F1 Score" ].plot( kind = "barh" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "F1 Score Accuracy of ML Perf ML Dataset",
                                                                                                                ax = axis[ 1 ] )
        

        Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()[ "AUC Score" ].plot( kind = "bar" ,
                                                                                                                figsize = ( 20 , 13 ),
                                                                                                                title = "AUC Score Accuracy of ML Perf ML Dataset",
                                                                                                                ax = axis[ 2 ] )
        
        plt.show()


    # Show the Insight of Analyzing and Testing ML Evaluation ML Dataset

    Analyze_and_Testing_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset.transpose()

    if ML_Key_Metrics == "Acc" :

        Best_ML_Accuracy_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset[ Analyze_and_Testing_ML_Perf_ML_Dataset[ "Accuracy" ] == Analyze_and_Testing_ML_Perf_ML_Dataset[ "Accuracy" ].max() ].index[ 0 ]

        Best_ML_Accuracy_ML_Perf = Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ]

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Accuracy ML Model is {} With Accuracy {} with F1 Score {} with AUC Score Accuracy {}".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ] ) ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "AUC Score" ] )))
            
        else :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics Accuracy ML Model is {} With Accuracy {} with F1 Score {} ".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ] ) ))
            
    elif ML_Key_Metrics == "F1 Score" :

        Best_ML_Accuracy_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset[ Analyze_and_Testing_ML_Perf_ML_Dataset[ "F1 Score" ] == Analyze_and_Testing_ML_Perf_ML_Dataset[ "F1 Score" ].max() ].index[ 0 ]

        Best_ML_Accuracy_ML_Perf = Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ]

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics F1 Score ML Model is {} With Accuracy {} with F1 Score {} with AUC Score Accuracy {}".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ] ) ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "AUC Score" ] )))
            
        else :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics F1 Score ML Model is {} With Accuracy {} with F1 Score {} ".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ] ) ))
            
    elif ML_Key_Metrics == "AUC Score" :

        Best_ML_Accuracy_ML_Perf_ML_Dataset = Analyze_and_Testing_ML_Perf_ML_Dataset[ Analyze_and_Testing_ML_Perf_ML_Dataset[ "AUC Score" ] == Analyze_and_Testing_ML_Perf_ML_Dataset[ "AUC Score" ].max() ].index[ 0 ]

        Best_ML_Accuracy_ML_Perf = Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "AUC Score" ]

        if is_ML_Column_Predicted_ML_Dataset_Binary_Classification :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics AUC Score ML Model is {} With Accuracy {} with F1 Score {} with AUC Score Accuracy {}".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ] ) ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "AUC Score" ] )))
            
        else :

            print( "------------------------------------------------------------------------------------------------------")
            print( "The Best Model for Key Metrics AUC Score ML Model is {} With Accuracy {} with F1 Score {} ".format( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "Accuracy" ] ),
                                                                                                                                                                str( Analyze_and_Testing_ML_Perf_ML_Dataset.loc[ Best_ML_Accuracy_ML_Perf_ML_Dataset ][ "F1 Score" ] ) ))
            
    # Train the Best ML Evaluation using ML Dataset ML Column Predicted
    
    X_train = ML_Dataset[ [ i for i in ML_Dataset.columns  if i != ML_Column_Predicted ]]
    
    #col = [ ( i + 2 )  for i in range( 5 ) if i % 2 == 0 ]

    y_train = ML_Dataset[ ML_Column_Predicted ]

    if Best_ML_Accuracy_ML_Perf_ML_Dataset == "KNN Neighbor Perf" :

        neigh = KNeighborsClassifier()

        neigh.fit( X_train, y_train )

        return( ( Best_ML_Accuracy_ML_Perf_ML_Dataset , 
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    neigh ) )
    
    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "SVM Perf" :
        
        clf = SVC()

        clf.fit( X_train , y_train )

        return( ( Best_ML_Accuracy_ML_Perf_ML_Dataset , 
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    clf ) )

    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "Random Forest Perf" :

        clf = RandomForestClassifier()

        clf.fit( X_train , y_train )

        return( ( Best_ML_Accuracy_ML_Perf_ML_Dataset , 
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    clf ) )
    
    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "Adaboost Perf" :

        clf = AdaBoostClassifier()

        clf.fit( X_train , y_train )

        return( ( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    clf ) )
    
    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "XGBoost Perf" :

        model = xgb.XGBClassifier()

        model.fit( X_train ,  y_train )

        return( ( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    model ) )

    elif Best_ML_Accuracy_ML_Perf_ML_Dataset == "Logistic Classifier Perf" :

        
        List_of_ML_Dataset_Scalar_and_Float_Column = X_train.select_dtypes( include=[float , int] )

        if len( List_of_ML_Dataset_Scalar_and_Float_Column ) > 1 :

            Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale = StandardScaler().fit( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] )

            X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_train[ List_of_ML_Dataset_Scalar_and_Float_Column ])

            X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] = Standardization_of_ML_Dataset_Scalar_and_Float_Column_Scale.transform( X_test[ List_of_ML_Dataset_Scalar_and_Float_Column ] ) 

        logreg = LogisticRegression()

        logreg.fit( X_train , y_train )

        return( ( Best_ML_Accuracy_ML_Perf_ML_Dataset ,
                    Best_ML_Accuracy_ML_Perf , 
                    Analyze_and_Testing_ML_Perf_ML_Dataset.transpose() ,
                    logreg ) )

    return None

        