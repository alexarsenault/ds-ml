
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re, csv

from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

def find_outliers(df_col, num_stds_away):
    col_stdev = df_col.std()
    col_mean = df_col.mean()

    outlier_indicies = df_col[(df_col>(col_mean + num_stds_away*col_stdev)) \
        | (df_col<(col_mean - num_stds_away*col_stdev))].index
    return outlier_indicies

def fill_in_values(df):
    quant_features = [f for f in df.columns if df.dtypes[f] != 'object']
    qual_features = [f for f in df.columns if df.dtypes[f] == 'object']

    for i in quant_features:
        df[i].fillna(df[i].median(), inplace=True)
    for i in qual_features:
        df[i].fillna("None", inplace=True)

    return df

def main():
    train_df = pd.read_csv('../data/housing_data/housing_train.csv')
    test_df = pd.read_csv('../data/housing_data/housing_test.csv')

    print(train_df.describe().T)

    # Create correlation matrix to see which features are correlated with sale price
    train_corr = train_df.corr()
    train_corr.sort_values(["SalePrice"], ascending = False, inplace = True)

    print(train_corr["SalePrice"])

    ax = plt.subplots(ncols=1, figsize=(10,10))
    corr_matrix = train_df.corr()
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_matrix, mask=mask, vmin = -1, vmax = 1, center = 0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation_matrix.png', dpi=300)
    plt.show()

    # explore variable pairs
    sns.boxplot(x=train_df["OverallQual"], y=train_df["SalePrice"])
    plt.title('Overall Quality vs. Sale Price')
    plt.xlabel('Overall Quality (1-10)')
    plt.ylabel('Sale Price ($)')
    plt.tight_layout()
    plt.savefig('qual_price_box.png', dpi=300)
    plt.show()

    # Find outliers of the features we want to use
    out_ind_oq = pd.Series(find_outliers(train_df["OverallQual"],4))
    out_ind_gla = pd.Series(find_outliers(train_df["GrLivArea"],4))
    out_ind_gc = pd.Series(find_outliers(train_df["GrLivArea"],4))
    out_ind_ga = pd.Series(find_outliers(train_df["GarageArea"],4))
    out_ind_tbs = pd.Series(find_outliers(train_df["TotalBsmtSF"],4))
    out_ind_1fs = pd.Series(find_outliers(train_df["1stFlrSF"],4))
    out_ind_fb = pd.Series(find_outliers(train_df["FullBath"],4))
    out_ind_trag = pd.Series(find_outliers(train_df["TotRmsAbvGrd"],4))
    out_ind_yb = pd.Series(find_outliers(train_df["YearBuilt"],4))
    out_ind_yr = pd.Series(find_outliers(train_df["YearRemodAdd"],4))
    out_ind_gyb = pd.Series(find_outliers(train_df["GarageYrBlt"],4))
    
    outlier_ind = pd.concat([out_ind_oq, out_ind_gla, out_ind_gc, out_ind_ga, out_ind_tbs, out_ind_1fs, \
                            out_ind_fb, out_ind_trag, out_ind_yb, out_ind_yr, out_ind_gyb])
    
    outlier_ind = outlier_ind.drop_duplicates()
    outlier_ind = pd.Index(outlier_ind)
    train_df = train_df.drop(outlier_ind)
    
    """
    for i in train_df.select_dtypes(include='object').columns:
        sns.boxplot(x=train_df[i], y = train_df['SalePrice'])
        plt.xticks(rotation=90)
        plt.show()
    """

    """
    Top features correlated with sale price:
        1. OverallQual - overall quality of the house
        2. GrLivArea - above grade living area
        3. GarageCars - size of garage in car capacity
        4. GarageArea - square footage of garage area
        5. TotalBsmtSF - basement square footage
        6. 1stFlrSF - square footage of first floor
        7. FullBath - full bathrooms above grade
        8. TotRmsAbvGrd - total rooms above grade
        9. YearBuilt - year house was built
        10. YearRemodAdd - remodel date (same as construction if no remodel)
        11. GarageYrBlt - year garage was built
    """

    # Separate out quanlitative and quantitative data
    train_df = fill_in_values(train_df)
    test_df = fill_in_values(test_df)

    train_df = pd.get_dummies(data=train_df)
    test_df = pd.get_dummies(data=test_df)

    train_df_use = train_df[["OverallQual", "GrLivArea", "GarageCars","GarageArea","TotalBsmtSF",\
                             "1stFlrSF", "FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd", "GarageYrBlt"]]
    
    test_df_use = test_df[["OverallQual", "GrLivArea", "GarageCars","GarageArea","TotalBsmtSF",\
                             "1stFlrSF", "FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd", "GarageYrBlt"]]
    test_df_use_Id = test_df["Id"]

    X_train, X_test, y_train, y_test = train_test_split(train_df_use, train_df['SalePrice'], test_size=0.25, random_state=10)

    # Search parameters
    n_estimators_gs = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features_gs = ['auto','sqrt']
    max_depth_gs = [int(x) for x in np.linspace(50, 200, num = 10)]
    max_depth_gs.append('None')
    min_samples_split_gs = [2,5,10]
    min_samples_leaf_gs = [1,2,4]

    features_grid = {'n_estimators': n_estimators_gs,
                     'max_features': max_features_gs,
                     'max_depth': max_depth_gs,
                     'min_samples_split': min_samples_split_gs,
                     'min_samples_leaf': min_samples_leaf_gs }

    rf = RandomForestRegressor(oob_score=True)
    grid = RandomizedSearchCV(rf, param_distributions=features_grid, n_iter=10, n_jobs=-1)
    y = train_df["SalePrice"]
    grid.fit(train_df_use,y)

    rf = grid.best_estimator_
    #prediction_pct = rf.oob_prediction_
    final_predictions = rf.predict(test_df_use)
    
    """
    Lasso model
    """
    Lasso_model = LassoCV(alphas = [1, 0.1, 0.01, 0.001, 0.0005], selection='random', max_iter=15000).fit(train_df_use, y)
    y_pred = Lasso_model.predict(train_df_use)

    print(r2_score(y, y_pred))
    print(np.sqrt(mean_squared_error(y,y_pred)))

    Lasso_Test = Lasso_model.predict(test_df_use)
    #coef = pd.Series(Lasso_model.coef_, index = x.columns)

    df = pd.DataFrame({'Actual': y.values.flatten(), 'Predicted': y_pred.flatten()})
    plt.scatter(y,y_pred)
    plt.show()

    # prepare data for enumerate
    lambdas = (0.5, 1, 2, 10, 100, 500, 1000)
    coeff_a = np.zeros((len(lambdas), train_df_use.shape[1]))
    train_r_squared = np.zeros(len(lambdas))
    test_r_squared = np.zeros(len(lambdas))

    # enumerate through lambdas with index and i
    for ind, i in enumerate(lambdas):    
        reg = Lasso(alpha=i)
        reg.fit(X_train, y_train)

        coeff_a[ind,:] = reg.coef_
        train_r_squared[ind] = reg.score(X_train, y_train)
        test_r_squared[ind] = reg.score(X_test, y_test)
    
    # Plotting
    plt.figure(figsize=(18, 8))
    plt.plot(train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
    plt.plot(test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
    plt.xlabel('Lamda index'); plt.ylabel(r'$R^2$')
    plt.xlim(0, len(lambdas) - 1)
    #plt.title(r'Evaluate lasso regression with lamdas: 0 = 0.001, 1= 0.01, 2 = 0.1, 3 = 0.5, 4= 1, 5= 2, 6 = 10, 7 = 100, 8 = 1000')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('lambda_lasso_effect.png', dpi=300)
    plt.show()
    
    # Write to output
    with open ('housing_results.csv', mode='w') as housing_results:
        housing_write = csv.writer(housing_results,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        housing_write.writerow(["Id", "SalePrice"])
        
        for i in range(len(final_predictions)):
            
            #housing_write.writerow([test_df_use_Id.iloc[i], Lasso_Test[i]])
            housing_write.writerow([test_df_use_Id.iloc[i], final_predictions[i]])

if __name__ == "__main__":
    main()