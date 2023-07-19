
class MetricsCompetition():
    def __init__(self, results: dict) -> None:
        self.results = results
        self.available_metrics = list(results.values())[0].keys() # Obtener las métricas disponibles
        self.best_model =  None
        self.best_metrics =  None


    def evaluated_best_model(self):
        # Evaluar los modelos y seleccionar el mejor
        self.best_metrics = self.results[list(self.results.keys())[0]]
        self.best_model = list(self.results.keys())[0]
        for name,metrics in self.results.items():
            total_score = 0
            if metrics["mse"] < self.best_metrics["mse"]: 
                total_score += 1 
            if metrics["mape"] < self.best_metrics["mape"]:
                total_score += 1 
            if metrics["r2"] > self.best_metrics["r2"]:
                total_score += 1

            if  total_score >= 2:
                self.best_metrics = metrics
                self.best_model =  name
            
        return self.best_model,self.best_metrics