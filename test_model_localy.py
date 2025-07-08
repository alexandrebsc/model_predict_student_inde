import joblib
import numpy as np

# individual test
# tranformed_input = np.array([[
#     15/30.,#age
#     4/7.,#level
#     5/10.,#ian
#     10/10.,#iaa
#     9.9/10.,#ieg
#     1. if False else 0.
# ],])

# train batch 2020 studants used to train  
# ALUNO-295, ALUNO-708, ALUNO-935, ALUNO-460, ALUNO-450, ALUNO-1067, ALUNO-1222
# tranformed_input = np.array([
#     [9/30,1/7,10/10,9.00/10,8.9/10,1.], #real 7.586 predicted 8.021
#     [15/30,4/7,5/10,7.91/10,6.4/10,1.], #real 6.957 predicted 6.723
#     [15/30,5/7,10/10,9.58/10,9./10,1.], #real 8.113 predicted 8.194
#     [10/30,1/7,5/10,8.50/10,8.1/10,1.], #real 8.004 predicted 7.719
#     [8/30,0/7,10/10,10.00/10,10./10,0.], #real 9.064 predicted 8.749
#     [9/30,0/7,5/10,8.50/10,10./10,1.], #real 8.312 predicted 7.784
#     [17/30,7/7,10/10,9.16/10,6.6/10,1.], #real 7.533 predicted 7.345
# ])

# test batch 2022 studants not used to train  
# ALUNO-608, ALUNO-1135, ALUNO-856, ALUNO-405, ALUNO-664, ALUNO-180, ALUNO-967, ALUNO-288, ALUNO-868
tranformed_input = np.array([
    [13/30,0/7,5.0/10.,9.000/10.,7.916/10.,1.],# real 6.039 predicted 6.98129126468761
    [14/30,4/7,10.0/10.,7.916/10.,8.724/10.,0.],# real 8.107 predicted 7.738901664154172
    [13/30,1/7,10.0/10.,8.500/10.,8.389/10.,0.],# real 7.011 predicted 7.7785254125998735
    [16/30,5/7,5.0/10.,7.083/10.,7.560/10.,0.],# real 6.576 predicted 7.158584571514944
    [17/30,6/7,5.0/10.,10.000/10.,9.485/10.,0.],# real 8.842 predicted 8.584660172385037
    [13/30,5/7,5.0/10.,9.166/10.,9.270/10.,1.],# real 7.641 predicted 7.786415286534241
    [13/30,0/7,5.0/10.,9.500/10.,8.888/10.,0.],# real 7.569 predicted 7.518181392983848
    [13/30,0/7,5.0/10.,9.000/10.,5.394/10.,0.],# real 6.994 predicted 6.1611354347701655
    [13/30,0/7,5.0/10.,8.500/10.,8.317/10.,1.],# real 7.369 predicted 7.09701654052131
])


model = joblib.load('random_forest_regressor_predict_student_inde.pkl')

for prediction in model.predict(tranformed_input):
    print(prediction*10)