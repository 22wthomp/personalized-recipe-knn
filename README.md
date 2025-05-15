# personalized-recipe-knn
# Personalized Recipe Recommender
This project is a recommendation app that uses K-Nearest Neighbor to recommend meals based on your nutritional prefrences/requirements. You set which nutrients matter to you and set a desired range using sliders and then get recipes that best match these preferences.

Built with Dash and Scikit-learn 

## How this works 
1. Choose which nutrients matter (Protein, Sugar, Sodium, etc)
2. Set slider ranges (like 30-50 grams of Protien)
3. KNN finds the 5 recipes closest to your nutritional preferences
4. Optionally see Mean Absolute Error and nutrient distribution with histograms

## Features
- interactive sliders and checkboxes for each nutrient
- KNN based matching
- Data Scaling with Standard Scaler
- Clean output table with recommended recipes

## Dataset
This project uses the **Food.com Recipes and Reviews** dataset from Kaggle:
[Food.com Recipes and Reviews – Kaggle Dataset](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)  
Author: Irkaal | Source: Kaggle
The dataset in this project was not collected by me. Used solely for educational non-commercial purposes 
Due to GitHub’s file size limits, you will need to manually download the dataset:

 **Download CSV**: [Google Drive Link](https://drive.google.com/file/d/1O3fiCI1sGaC7CkO_j9nt-BE-ZG2RTkWj/view?usp=drive_link)
 Place in folder titled: data/recipes.csv

## Running the App
1. Clone this repo:
   git clone https://github.com/yourusername/Recipe_Recommender_Will_Thompson.git
   cd Recipe_Recommender_Will_Thompson
2. Install Libraries
   pip install dash pandas scikit-learn plotly
3. Make sure recipes.csv is in data/folder
4. Run App
5. Open  http://127.0.0.1:8050/ in browser

## About
Built as Final Project for CS-437 (Machine Learning) at SIUC Created by Will Thompson
