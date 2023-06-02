# Class Omlette Object -- Automatic Analyze and Testing ML Models
# Function Show Analyze and Testing ML Model and Show Analyze and Testing ML Models Feature Importance to Show Analyze and Testing ML Model and Show Analyze and Testing ML Models Feature Impotance
# Created by : Christofel Rio Goenawan
# AI & Analytics Consultant at Hyundai and AI & Robotics Master Student at Korean Advanced Institute of Science and Technology
# If there Are A lot of Advices and Want to Discuss Automatic Analyzing and Testing ML Models Omlette Kindly Chit Chat Me on christofel.goenawan@kaist.ac.kr


import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# Python Package for Check All Package Has Been Installed or Not

from Check_All_Package_Has_Been_Installed_or_Not import Check_All_Package_Has_Been_Installed_or_Not

# Python Package for Analyze and Testing ML Model for ML Model Regression Task and ML Model Classification Task

from Analyze_and_Testing_ML_Model import Analyze_and_Testing_ML_Model_for_Classification , Analyze_and_Testing_ML_Model_for_Regression


class Omlette( ):

    def __init__( self ):

        print( "Welcome to AutOmatic ML Tester -- Omlette \n\
                Automatic ML Tester Helps Data Scientist and ML Engineer to Automatically Analyzing and Testing Various ML Model to Analyze and Fit Dataset" )

        # First Check All Needed Python Package Has Been Installed or Not

        print( "--------------------------------------------------------------------------------")

        print( "Checking All Needed Python Package Has Been Installed or Not... " )

        is_All_Needed_Package_Has_Installed_Installed = Check_All_Package_Has_Been_Installed_or_Not()

        if not is_All_Needed_Package_Has_Installed_Installed :
        
            print( "There are Some Needed ML Package Has Not Been Installed... Please Installed Needed ML Package :-)" )

            return None
        
        self.ML_Dataset = None

        self.ML_Column_Predicted = None

        self.ML_Task = None

        self.ML_Best_ML_Perf_Testing_Result = None
        
        self.ML_Key_Metrics_ML_Testing_Result = None

        self.ML_Result = None

        self.ML_Best_ML = None
            

    def Analyze_and_Testing_ML_Model( self ,
                                     ML_Dataset : pd.DataFrame ,
                                     ML_Column_Predicted : str ,
                                     ML_Task : str  = "Regr" ,
                                     ML_Key_Metrics : str = None ,
                                     ML_Evaluation_Cross_Validation_Number_Cross_Validation : int = 5 ) :

        ML_Dataset = pd.DataFrame( ML_Dataset )
        
        self.ML_Dataset = ML_Dataset

        self.ML_Column_Predicted = ML_Column_Predicted
        
        self.ML_Task = ML_Task
        
        if ML_Task == "Regr" :
        
            ( self.ML_Best_ML_Perf_Testing_Result ,
            self.ML_Key_Metrics_ML_Testing_Result ,
            self.ML_Result ,
            self.ML_Best_ML ) = Analyze_and_Testing_ML_Model_for_Regression(  ML_Dataset , ML_Column_Predicted  , ML_Key_Metrics , ML_Evaluation_Cross_Validation_Number_Cross_Validation )

        elif ML_Task == "Classif" : 
        
            ( self.ML_Best_ML_Perf_Testing_Result ,
            self.ML_Key_Metrics_ML_Testing_Result ,
            self.ML_Result ,
            self.ML_Best_ML ) = Analyze_and_Testing_ML_Model_for_Classification( ML_Dataset , ML_Column_Predicted , ML_Key_Metrics , ML_Evaluation_Cross_Validation_Number_Cross_Validation )

        else :
            
            print( "Please Setting ML Task... ML Task 'Regr' for ML Task Regression and ML Task 'Classif' for ML Task Classification" )

            return
        

        return( { "ML_Best_Perf_Testing_Result" : self.ML_Best_ML_Perf_Testing_Result , 
                 "ML_Key_Metrics_ML_Result" : self.ML_Key_Metrics_ML_Testing_Result , 
                 "ML_Result" : self.ML_Result ,
                 "ML_Best_ML" :  self.ML_Best_ML } )
    
    
    def Show_Analyze_and_Testing_ML_Result( self ) :

        if self.ML_Result is None :
            # If there is No ML Result

            print( "There is No ML Result... Please Analyze and Testing ML Model to Analyze and Testing ML Model" )

            return None
        
        else :

            if self.ML_Task == "Classif" :

                sns.set_theme()

                Analyzing_and_Testing_ML_Perf_ML_Dataset = self.ML_Result

                if "AUC Score" not in Analyzing_and_Testing_ML_Perf_ML_Dataset.index :

                    figure, axis = plt.subplots( 2 ,  1 )

                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "Accuracy" ].plot( kind = "bar" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Accuracy of ML Perf ML Dataset",
                                                                                                                            ax = axis[ 0 ] )
                    
                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "F1 Score" ].plot( kind = "barh" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "F1 Score Accuracy of ML Perf ML Dataset",
                                                                                                                            ax = axis[  1 ] )
                    

                    plt.show()

                else :

                    figure, axis = plt.subplots( 3 ,  1 )

                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "Accuracy" ].plot( kind = "bar" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Accuracy of ML Perf ML Dataset",
                                                                                                                            ax = axis[ 0 ] )
                    
                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "F1 Score" ].plot( kind = "barh" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "F1 Score Accuracy of ML Perf ML Dataset",
                                                                                                                            ax = axis[ 1 ] )
                    

                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "AUC Score" ].plot( kind = "bar" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "AUC Score Accuracy of ML Perf ML Dataset",
                                                                                                                            ax = axis[ 2 ] )
                    
                    plt.show()
            
            elif self.ML_Task == "Regr" :

                sns.set_theme()

                Analyzing_and_Testing_ML_Perf_ML_Dataset = self.ML_Result

                if "RMSLE" not in Analyzing_and_Testing_ML_Perf_ML_Dataset.index : 

                    figure, axis = plt.subplots( 2 ,  1 )

                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "RMSE" ].plot( kind = "bar" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Root Mean Square Error of ML Perf ML Dataset",
                                                                                                                            ax = axis[  0 ] )
                    
                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "MAE" ].plot( kind = "barh" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Mean Absolute Error of ML Perf ML Dataset",
                                                                                                                            ax = axis[ 1 ] )
                    

                    plt.show()

                else :

                    figure, axis = plt.subplots( 3 ,  1 )

                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "RMSE" ].plot( kind = "bar" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Root Mean Square Error of ML Perf ML Dataset",
                                                                                                                            ax = axis[  0 ] )
                    
                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "MAE" ].plot( kind = "barh" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Mean Absolute Error of ML Perf ML Dataset",
                                                                                                                            ax = axis[  1 ] )
                    

                    Analyzing_and_Testing_ML_Perf_ML_Dataset.transpose()[ "RMSLE" ].plot( kind = "bar" ,
                                                                                                                            figsize = ( 20 , 13 ),
                                                                                                                            title = "Root Mean Square Logistic Regression Error of ML Perf ML Dataset",
                                                                                                                            ax = axis[  2 ] )
                    
                    plt.show()
            
            return None

    
    def Show_Analyze_and_Testing_ML_Feature_Importance( self ) :

        print( "------------------------------------------------------------------------")

        print( "The best ML Best ML Perf Testing Result is : " + str( self.ML_Best_ML_Perf_Testing_Result ) )

        print( "-------------------------------------------------------------------------" )

        if self.ML_Best_ML_Perf_Testing_Result is None :

            print( "There are No ML Best ML Perf Testing Result... Please Analyze and Testing ML Dataset..." )

            return None
        
        elif self.ML_Best_ML_Perf_Testing_Result == "Random Forest Perf" :
            # If ML Best ML Perf Testing Result is Random Forest Perf

            model = self.ML_Best_ML

            feat_importances = pd.Series( model.feature_importances_, index= [ i for i in self.ML_Dataset.columns if i != str( self.ML_Column_Predicted ) ] )

            print( feat_importances )

            sns.set_theme()

            figure , axis = plt.subplots( figsize = ( 20 , 13 ) )

            feat_importances.nlargest( 10 ).plot(kind='barh' , ax = axis , title = "Top 10 Feature Importance of ML Best ML Perf Testing Result " )

            plt.show()

        elif self.ML_Best_ML_Perf_Testing_Result == "Adaboost Perf" :
            # If ML Best ML Perf Testing Result is Adaboost Perf

            model = self.ML_Best_ML

            feat_importances = pd.Series( model.feature_importances_, index= [ i for i in self.ML_Dataset.columns if i != str( self.ML_Column_Predicted ) ] )

            print( feat_importances )

            sns.set_theme()

            figure , axis = plt.subplots( figsize = ( 20 , 13 ) )

            feat_importances.nlargest( 10 ).plot(kind='barh' , ax = axis , title = "Top 10 Feature Importance of ML Best ML Perf Testing Result " )

            plt.show()

        elif self.ML_Best_ML_Perf_Testing_Result == "XGBoost Perf" :
            # If ML Best ML Perf Testing Result is XGBoost Perf

            model = self.ML_Best_ML

            figure , ax = plt.subplots(figsize=(12,30))

            import xgboost as xgb

            xgb.plot_importance(model, max_num_features=10, height=1, ax=ax)

            ax.tick_params(axis='y', which='both', labelsize=25)

            ax.tick_params(axis='x', which='both', labelsize=25)

            plt.show()

        elif self.ML_Best_ML_Perf_Testing_Result == "Logistic Classifier Perf" :
            # If ML Best ML Perf Testing Result is Logistic Classifier Perf

            model = self.ML_Best_ML

            coefficients = model.coef_[0]

            feature_importance = pd.DataFrame({'Feature': [ i for i in self.ML_Dataset.columns if i != self.ML_Column_Predicted], 'Importance': np.abs(coefficients)})
            feature_importance = feature_importance.sort_values('Importance', ascending=True).head( 10 )
            
            print( feature_importance.to_string() )

            figure , axis = plt.subplots( figsize = ( 20 , 13 ))

            feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6) , ax = axis )

            plt.show()

        elif self.ML_Best_ML_Perf_Testing_Result == "Linear Regression Perf"  :
            # If ML Best ML Perf Testing Result is Linear Regression Perf

            model = self.ML_Best_ML

            coefficients = model.coef_[0]

            feature_importance = pd.DataFrame({'Feature': [ i for i in self.ML_Dataset.columns if i != self.ML_Column_Predicted], 'Importance': np.abs(coefficients)})
            feature_importance = feature_importance.sort_values('Importance', ascending=True).head( 10 )
            
            print( feature_importance.to_string() )

            figure , axis = plt.subplots( figsize = ( 20 , 13 ))

            feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6) , ax = axis )

            plt.show()
        
        return figure
    
            
        