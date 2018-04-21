class stepwise_aic_classs():
        
    def step_aic(df, exp_v, obj_v,test_size=0.3, random_state=0, **kwargs):
        from sklearn.cross_validation import train_test_split
        from sklearn.linear_model import LinearRegression
        from AIC import AIC_class
        import numpy as np
        
        '''
        Args:
            exp_v: Explanatory variables
            #obj_v (str or list): objective variable
            #kwargs: extra keyword argments for model (e.g., data, family)
            
        #Returns:
            #model: a model that seems to have the smallest AIC
        '''
                
        formula_head = ' + '.join(obj_v) + ' ~ '
        formula = formula_head + '1 + ' + ' + '.join(exp_v)
        X = df[exp_v]
        y = df[obj_v]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        slr = LinearRegression()
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)
        aic = AIC_class.AIC(y_train, y_train_pred, slr)

        print('Initial AIC: {}'.format(round(aic, 3)))
        print('Initial formula: {}'.format(formula))
        current_score, best_new_score = np.ones(2) * aic
        #print(current_score)
        original_variable = set(exp_v)
        remaining = original_variable.copy()
        
        while remaining and current_score == best_new_score:
            
            scores_with_candidates = []
            
            for candidate in original_variable:
                remaining = original_variable.copy()
                remaining.remove(candidate)
                X_temp = df[np.array(list(remaining))]
                y_temp = df[obj_v]
                X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_state)
                slr = LinearRegression()
                slr.fit(X_train_temp, y_train_temp)
                y_train_temp_pred = slr.predict(X_train_temp)
                y_test_temp_pred = slr.predict(X_test_temp)
                aic_temp = AIC_class.AIC(y_train_temp, y_train_temp_pred, slr)
                #aic_temp = math.sqrt(mean_squared_error(y_train_temp, y_train_temp_pred))
                #formula_temp = formula_head + '1 + ' + ' + '.join(remaining)
                #print('AIC: {}, formula: {}'.format(round(aic_temp, 3), formula_temp))
                scores_with_candidates.append((aic_temp, candidate))
        
            scores_with_candidates.sort()
            scores_with_candidates.reverse()
            #print(scores_with_candidates)
            best_new_score, best_candidate = scores_with_candidates.pop()
            
            #print(best_new_score, best_candidate)

            if best_new_score < current_score:
                original_variable.remove(best_candidate)
                #selected.append(best_candidate)
                current_score = best_new_score
                
            formula_best = formula_head + '1 + ' + ' + '.join(original_variable)
            print('Updated AIC: {}'.format(current_score))
            print('Updated formula: {}'.format(formula_best))

            #print(original_variable)

        #formula = formula_head + ' + '.join(original_variable)
        #print('The best formula: {}'.format(formula))
        formula_head = ' + '.join(obj_v) + ' ~ '
        formula = formula_head + '1 + ' + ' + '.join(original_variable)
        X = df[np.array(list(original_variable))]
        y = df[obj_v]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        slr = LinearRegression()
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)
        aic = AIC_class.AIC(y_train, y_train_pred, slr)

        print('Final AIC: {}'.format(round(aic, 3)))
        print('Final formula: {}'.format(formula))
        
        return y_train_pred, y_test_pred, X_train, X_test, y_train, y_test #print('The best formula: {}'.format(formula))
    
    
    