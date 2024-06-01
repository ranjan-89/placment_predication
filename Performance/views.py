from django.shortcuts import render
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
def index(request):
    context = {
        'welcome_message': "Welcome to Placement Prediction System"
    }
    return render(request, 'index.html', context)

def result(request):
    if request.method == 'POST':
        model = joblib.load('/home/kira/Documents/Placement_Prediction/Placement.joblib')
        
        # Extract form data
        cgpa = float(request.POST.get('cgpa', 0))  # default value in case of conversion failure
        internships = int(request.POST.get('internships', 0))
        projects = int(request.POST.get('projects', 0))
        workshopscertifications = int(request.POST.get('workshopsCertifications', 0))
        aptitudeTestScore = int(request.POST.get('aptitudeTestScore', 0))
        softSkillsRating = float(request.POST.get('softSkillsRating', 0))

        activity = request.POST.get('extracurricularActivities', 0)
        if activity == "yes":
            activity=1
        elif activity ==  "no":
            activity=0
        extracurricularActivities = activity

        training = request.POST.get('placementTraining', 0)
        if training=="yes":
            training=1
        elif training == "no":
            training=0
        placementTraining = training

        sscMarks = int(request.POST.get('sscMarks', 0))
        hscMarks = int(request.POST.get('hscMarks', 0))

        columns = ['cgpa', 'internships', 'projects', 'workshopscertifications', 'aptitudeTestScore', 
                'softSkillsRating', 'extracurricularActivities', 'placementTraining', 'sscMarks', 'hscMarks']
        data = [[cgpa, internships, projects, workshopscertifications, aptitudeTestScore, 
                            softSkillsRating, extracurricularActivities, placementTraining, sscMarks, hscMarks]]

        data_df = pd.DataFrame(data, columns=columns)

        # Apply LabelEncoder to categorical columns
        lb = LabelEncoder()
        data_df['extracurricularActivities'] = lb.fit_transform(data_df['extracurricularActivities'])
        data_df['placementTraining'] = lb.fit_transform(data_df['placementTraining'])

        
        # Scale numerical features
        std = StandardScaler()
        data = std.fit_transform(data_df)

        prediction = model.predict(data)
        

        return render(request, 'result.html', {'prediction': prediction})

    context = {
        'Result': "Sorry! Can't load at that time"
    }
    return render(request, 'index.html', context)