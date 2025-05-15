import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import os
#Dataset from Kaggle (not collected by author)
try:
    recipes = pd.read_csv(r"data/recipes.csv")
except FileNotFoundError:
    raise FileNotFoundError("recipes.csv not found. please download the dataset and place it in the 'data/' folder.")

# list of nutrients that matter
nutrients = [
    'Calories',
    'ProteinContent',
    'CarbohydrateContent',
    'SodiumContent',
    'FatContent',
    'SaturatedFatContent',
    'CholesterolContent',
    'SugarContent'
]

# ranges for slider values
nutrient_ranges = {
    'Calories': {'min': 0, 'max': 1000, 'step': 10},
    'ProteinContent': {'min': 0, 'max': 100, 'step': 1},
    'CarbohydrateContent': {'min': 0, 'max': 300, 'step': 5},
    'SodiumContent': {'min': 0, 'max': 5000, 'step': 50},
    'FatContent': {'min': 0, 'max': 100, 'step': 1},
    'SaturatedFatContent': {'min': 0, 'max': 50, 'step': 1},
    'CholesterolContent': {'min': 0, 'max': 500, 'step': 5},
    'SugarContent': {'min': 0, 'max': 300, 'step': 5},
}

# ----------------Functions----------------


def removePT(df, columnName):
    #removes pt from time and add readability
    df[columnName] = df[columnName].str.replace('PT', '', regex=False)
    df[columnName] = df[columnName].str.replace('H', ' Hour ', regex=False)
    df[columnName] = df[columnName].str.replace('M', ' Minutes ', regex=False)
    return df


def changeNames(df, columnName):
    # adds readability to columns
    match columnName:
        case "CookTime":
            df = removePT(df, columnName)
            df.rename(columns={columnName: 'Cook Time'}, inplace=True)
            return df
        case "PrepTime":
            df = removePT(df, columnName)
            df.rename(columns={columnName: 'Prep Time'}, inplace=True)
            return df
        case "TotalTime":
            df = removePT(df, columnName)
            df.rename(columns={columnName: 'Total Time'}, inplace=True)
            return df
        case "RecipeIngredientParts":
            df[columnName] = df[columnName].str.replace(r'^c\(|\)$', '', regex=True)
            df[columnName] = df[columnName].str.replace('"', '').str.replace("'", "")
            df.rename(columns={columnName: 'Ingredients'}, inplace=True)
            return df
        case "AuthorName":
            df.rename(columns={columnName: 'Author'}, inplace=True)
            return df
        case "FatContent":
            df.rename(columns={columnName: 'Fat'}, inplace=True)
            return df
        case "SaturatedFatContent":
            df.rename(columns={columnName: 'Saturated Fat'}, inplace=True)
            return df
        case "CholesterolContent":
            df.rename(columns={columnName: 'Cholesterol'}, inplace=True)
            return df
        case "SodiumContent":
            df.rename(columns={columnName: 'Sodium'}, inplace=True)
            return df
        case "CarbohydrateContent":
            df.rename(columns={columnName: 'Carbohydrates'}, inplace=True)
            return df
        case "SugarContent":
            df.rename(columns={columnName: 'Sugar'}, inplace=True)
            return df
        case "ProteinContent":
            df.rename(columns={columnName: 'Protein'}, inplace=True)
            return df
        case _:
            return df

def preprocessData(recipes, nutrients):
    # copy to avoid modifying original data
    recipesOriginal = recipes.copy()
    # make sure that nutrients are numeric
    for nutrient in nutrients:
        recipes[nutrient] = pd.to_numeric(recipes[nutrient], errors='coerce')
    # Drop rows with missing nutrient values
    recipes.dropna(subset=nutrients, inplace=True)
    # apply scalar to dataset
    scaler = StandardScaler()
    recipesScaled = recipesOriginal.copy()
    recipesScaled[nutrients] = scaler.fit_transform(recipesScaled[nutrients])
    
    return recipesScaled, recipesOriginal, scaler

def scale_slider_values(sliderValues, nutrients, scaler):
    # function to scale slider values
    scaledSliderValues = []
    for i, nutrient in enumerate(nutrients):
        min_val, max_val = sliderValues[i]
        # Use the scaler's mean and scale 
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
        # Scale the min and max values
        scaled_min = (min_val - mean)/ scale
        scaled_max = (max_val - mean)/ scale
        scaled_min, scaled_max = min(scaled_min, scaled_max), max(scaled_min, scaled_max)
        scaledSliderValues.append([scaled_min, scaled_max])
    return scaledSliderValues

def nutritionConstraints(df, nutrients, checklist_values, scaled_slider_values):
    # applies the constraints to included nutrients
    filteredDF = df.copy()
    for i, nutrient in enumerate(nutrients):
        # Apply filter only if 'include' is checked
        if checklist_values[i] and 'include' in checklist_values[i]:
            min_val, max_val = scaled_slider_values[i]
            filteredDF = filteredDF[
                (filteredDF[nutrient] >= min_val) & (filteredDF[nutrient] <= max_val)
            ]
    return filteredDF

def userPreferences(scaledSliderValues, checklist_values, nutrients):
    # finds the mean of the included nutrient slider values
    preferences = []
    for i, (min_val, max_val) in enumerate(scaledSliderValues):
        if checklist_values[i] and 'include' in checklist_values[i]:
            middle = (min_val + max_val)/ 2
        else:
            middle = 0  # Neutral value for excluded nutrients
        preferences.append(middle)
    return np.array(preferences)

def getFeatureMatrix(constraintRecipes, nutrients, checklist_values):
    # creates feature matrix of nutrients that are included
    included_nutrients = [
        nutrient for nutrient, checklist in zip(nutrients, checklist_values)
        if checklist and 'include' in checklist
    ]
    X = constraintRecipes[included_nutrients].values
    return X, included_nutrients

def evaluate_distances(distances):
    # evaluate mean distance
    mean_distance = np.mean(distances)
    return mean_distance


def evaluate_mae(preferences, recommendedRecipes, included_nutrients):
    #  evaluate Mean Absolute Error 
    errors = []
    for nutrient in included_nutrients:
        recommended_values = recommendedRecipes[nutrient].values
        errors.append(np.abs(recommended_values - preferences[included_nutrients.index(nutrient)]).mean())
    return errors

def percentDifference(preferences, mae, included_nutrients):
    # function to calculate percent difference
    percent_diffs = []
    for i, nutrient in enumerate(included_nutrients):
        pref_value = preferences[i]
        error = mae[i]
        if pref_value != 0:
            percent_diff = (error / abs(pref_value)) * 100
        else:
            percent_diff = 0  
        percent_diffs.append(percent_diff)
    return percent_diffs

def outputTable(recipes, recommendedRecipes):
    # returns output table
    recommended = recipes[recipes['RecipeId'].isin(recommendedRecipes)]
    
    # columns that need to be removed
    columnsToRemove = ['RecipeIngredientQuantities','RecipeYield','RecipeInstructions','DatePublished','RecipeCategory','Keywords','ReviewCount','AuthorId','Description','FiberContent','AggregatedRating','RecipeServings','Images']
    droppedDf = recommended.drop(columns=columnsToRemove, axis=1)
    
    # Columns that need some editting to improve readability
    ptRemoveColumns = ['ProteinContent','SugarContent','CarbohydrateContent','SodiumContent','CholesterolContent','SaturatedFatContent','FatContent','AuthorName','CookTime','PrepTime','TotalTime','RecipeIngredientParts']
    for col in ptRemoveColumns:
        droppedDf = changeNames(droppedDf, col)
    #converts to dictionary for ease of puting it in a dash table
    myData = droppedDf.to_dict('records')
    columns = [{"name": col, "id": col} for col in droppedDf.columns]
    
    return dash_table.DataTable(
        data=myData,
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'margin-top': '20px',
            'fontWeight': 'bold'
        },
        sort_action='none',
        filter_action='none',       
        editable=False,             
        row_deletable=False,         
        page_action='none',          
    )

def create_histogram(recipes, nutrient):
    # Function to create a histogram for the nutrient that gets selected in drop down menu
    #max values for histgram
    nutrient_max_values = {
        'Calories': 1000,
        'ProteinContent': 80,
        'CarbohydrateContent': 200,
        'SodiumContent': 3000,
        'FatContent': 100,
        'SaturatedFatContent': 100,
        'CholesterolContent': 300,
        'SugarContent': 80,
    }

    data = recipes[nutrient].dropna()

    # Remove outliers beyond the maximum value
    max_value = nutrient_max_values.get(nutrient, None)
    if max_value is not None:
        data = data[data <= max_value]
        
    counts, bins = np.histogram(data, bins=30)

    # finds center of each bin for ploting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    #creates historgram
    fig = go.Figure(data=[
        go.Bar(
            x=bin_centers,
            y=counts,
            width=np.diff(bins),
            marker=dict(color='blue'),
            opacity=0.75
        )
    ])

    # Update layout
    fig.update_layout(
        title=f"Distribution of {nutrient}",
        xaxis_title=nutrient,
        yaxis_title="Frequency",
        xaxis=dict(range=[0, max_value]),
        template="plotly_white"
    )

    return fig

# ----------------Initialization/Prepare data-------------
recipesScaled, recipesOriginal, scaler = preprocessData(recipes, nutrients)

# ----------------Start of Dash app------------
app = Dash(__name__)

# Create sliders for each unit and include checkbox
nutrient_controls = []

for nutrient in nutrients:
    # Assigns nutrient min, max, and step from the dictionary nutrient_ranges
    nutrient_min = nutrient_ranges[nutrient]['min']
    nutrient_max = nutrient_ranges[nutrient]['max']
    nutrient_step = nutrient_ranges[nutrient]['step']
    control = html.Div([
        html.Label(nutrient),
        # Checkbox for include
        dcc.Checklist(
            id=f'{nutrient}-checklist',
            options=[{'label': 'Include', 'value': 'include'}],
            value=[],  # Start unchecked
            style={'margin-bottom': '5px'}
        ),
        # create sliders
        dcc.RangeSlider(
            id=f'{nutrient}-slider',
            min=nutrient_min,
            max=nutrient_max,
            step=nutrient_step,
            value=[nutrient_min, nutrient_max],
            marks={i: str(i) for i in range(nutrient_min, nutrient_max + 1, max(1, (nutrient_max - nutrient_min) // 5))},
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False,
            disabled=True  
        ),
    ], style={'margin-bottom': '20px'})
    nutrient_controls.append(control)

# layout of app
app.layout = html.Div([
    html.H1("Recipe Recommendation"),
    html.Div(nutrient_controls),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div([
        dcc.Checklist(
            id='additional-options',
            options=[
                {'label': 'Show Error Metrics', 'value': 'show-metrics'},
                {'label': 'Show Histogram', 'value': 'show-histogram'}
            ],
            value=[],  
            labelStyle={'display': 'inline-block', 'margin-right': '20px'}
        ),
    ], style={'margin-top': '20px'}),
    html.Div(id='output-container'),
    html.Div(
        id='histogram-container', 
        style={'display': 'none'},  
        children=[
            html.Label("Select Nutrient for Histogram"),
            dcc.Dropdown(
                id='nutrient-dropdown',
                options=[{'label': nutrient, 'value': nutrient} for nutrient in nutrients],
                value=nutrients[0],  
                clearable=False
            ),
            dcc.Graph(id='histogram-graph')  
        ]
    )
])

# ----------------Callbacks-----------------

# callback to toggle slider disabled state based on include checkbox
@app.callback(
    [Output(f'{nutrient}-slider', 'disabled') for nutrient in nutrients],
    [Input(f'{nutrient}-checklist', 'value') for nutrient in nutrients]
)
def toggle_sliders(*checklist_values):
    disabled_states = []
    for checklist in checklist_values:
        if checklist and 'include' in checklist:
            disabled_states.append(False) 
        else:
            disabled_states.append(True)
    return disabled_states

# callback for main function
@app.callback(
    [Output('output-container', 'children'),
     Output('histogram-container', 'style')],
    Input('submit-button', 'n_clicks'),
    State('additional-options', 'value'),
    [State(f'{nutrient}-checklist', 'value') for nutrient in nutrients] +
    [State(f'{nutrient}-slider', 'value') for nutrient in nutrients]
)
def update_output(n_clicks, additional_options, *args):
    if n_clicks > 0:
        # 1a. Split the args into checklist values and slider values
        checklistValues = args[:len(nutrients)] 
        sliderValues = args[len(nutrients):]
        
        # 1b.Scale the users slider values
        scaled_slider_values = scale_slider_values(sliderValues, nutrients, scaler)
        # 2. Apply user-defined filters on the scaled data
        constraintRecipesScaled = nutritionConstraints(recipesScaled, nutrients, checklistValues, scaled_slider_values)
        
        if constraintRecipesScaled.empty:
            return "No recipes match your nutritional requirements.", {'display':'none'}
        
        # 3. Calculate user preferences based on scaled slider values and inclusion
        preferences = userPreferences(scaled_slider_values, checklistValues, nutrients)
        
        # 4. Prepare feature matrix with only included nutrients
        X, includedNutrients = getFeatureMatrix(constraintRecipesScaled, nutrients, checklistValues)
        
        if len(includedNutrients) == 0:
            return "No nutrients selected for filtering. Please include at least one nutrient.", {'display':'none'}
        
        # 5. Instance of NearestNeighbors
        n_neighbors = 5  # Number of recommendations to make
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
        
        #6. Fit the model with the feature matrix
        knn.fit(X)
        
        # 7. Align preferences with included nutrients
        preferencesIncluded = preferences[[nutrients.index(nutrient) for nutrient in includedNutrients]]
        
        # 8. Find nearest neighbors
        distances, indices = knn.kneighbors([preferencesIncluded])
        
        # 9. Recommended recipes
        recommendedRecipes = constraintRecipesScaled.iloc[indices[0]]
        
        # Display the recommendations
        recommendedIds = recommendedRecipes['RecipeId'].tolist()
        
        # Output table
        output = outputTable(recipesOriginal, recommendedIds)
        output_list = [output]
        
        # if show metrics checked then 
        if 'show-metrics' in additional_options:
            # error metrics
            mean_distance = evaluate_distances(distances[0])
            mae = evaluate_mae(preferencesIncluded, recommendedRecipes[includedNutrients], includedNutrients)
            percentDiff = percentDifference(preferencesIncluded, mae, includedNutrients)
            averagePercentDiff = np.mean(percentDiff)
            error_metrics_div = html.Div([
                html.H3("Error Metrics"),
                html.P(f"Mean Distance to Recommendations: {mean_distance:.4f}"),
                html.P("Mean Absolute Error per Nutrient:"),
                html.Ul([html.Li(f"{nutrient}: {error:.4f}") for nutrient, error in zip(includedNutrients, mae)]),
                html.P("Percent Difference from Requirements:"),
                html.Ul([html.Li(f"{nutrient}: {error:.2f}%") for nutrient, error in zip(includedNutrients, percentDiff)]),
                html.P(f"Average Percent Difference: {averagePercentDiff:.2f}%"),
            ], style={'margin-top': '20px'})
            
            output_list.append(error_metrics_div)
        
        # if show historgram checked
        if 'show-histogram' in additional_options:
            histogram_style = {'display': 'block'}
        else:
            histogram_style = {'display': 'none'}
        
        return html.Div(output_list), histogram_style
    else:
        return "", {'display':'none'}

# Callback to update the histogram graph
@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('nutrient-dropdown', 'value'),
     Input('additional-options', 'value')] 
)
def update_histogram(selected_nutrient, additional_options):
    if 'show-histogram' in additional_options:
        fig = create_histogram(recipesOriginal, selected_nutrient)
        return fig
    else:
        return {}

# ----------------Run the app-----------------
if __name__ == '__main__':
    app.run_server(debug=True)