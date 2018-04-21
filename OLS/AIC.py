class AIC_class():
    def AIC(y, yhat, model):
        from sklearn import metrics
        import math
        aic = len(y)*math.log(2*math.pi*((y - yhat)**2).sum()/len(y)) + len(y) + 2*(model.coef_).size
        return aic