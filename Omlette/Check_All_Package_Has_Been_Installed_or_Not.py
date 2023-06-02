# Function Check_All_Package_Has_Been_Installed_or_Not() for Automatic Analyzing and Testing Machine Learning Models Omlette
# Function Check_All_Package_Has_Been_Installed_or_Not() to Check All Needed Python Package Has Been Installed Or Not
# Created by : Christofel Rio Goenawan
# AI & Analytics Consultant at Hyundai and AI & Robotics Master Student at Korean Advanced Institute of Science and Technology
# If there Are A lot of Advices and Want to Discuss Automatic Analyzing and Testing Machine Learning Models Omlette Kindly Chit Chat Me on christofel.goenawan@kaist.ac.kr

# List of All Python Package Needed by Automatic Analyzing and Testing Machine Learning Models Omlette


list_of_All_Python_Package_Needed = [ "pandas" , 
                                     "numpy" ,
                                     "seaborn" ,
                                     "sklearn" ,
                                     "xgboost" ]


def Check_All_Package_Has_Been_Installed_or_Not() :

    # First All Needed Package Has Not Been Installed
    
    is_All_Package_Has_Been_Installed_or_Not = False

    new_module = __import__( "pandas" )

    for Needed_Package_to_Installed in list_of_All_Python_Package_Needed :

        try :


            new_module = __import__( Needed_Package_to_Installed )

        
        except :
        # If The Needed Package to Installed Has Not Been Installed

            print( "There is Needed Python Package Has Not Been Installed .... ")

            print( "Python Package " + str( Needed_Package_to_Installed ) + " has Not Been Installed... " )

            print( "Please Install Needed Python Package " + str( Needed_Package_to_Installed ) + " ...... " )

            is_All_Package_Has_Been_Installed_or_Not = False

            return( is_All_Package_Has_Been_Installed_or_Not )
        
    
    # If All Needed Package Has Been Installed 
    
    is_All_Package_Has_Been_Installed_or_Not = True

    print( "All Needed Python Package Has Been Installed :-) .... " )

    return( is_All_Package_Has_Been_Installed_or_Not )



