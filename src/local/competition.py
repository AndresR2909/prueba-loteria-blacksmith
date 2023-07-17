
class MetricsCompetition():
    def __init__(self, results: dict, type: str) -> None:
        self.results = results
        self.available_metrics = list(results.values())[0].keys() # Obtener las métricas disponibles
        self.best_model =  None
        self.best_metric_score = 0
        self.best_total_score = 0
        self.type = type


    
    def _count_best_metrics(self,model):
        """Metodo para contar la cantidad de métricas superiores o iguales de modelo con respecto a los otros modelos"""
        #contar el modelo que tenga mas metricas mayores con respecto a los otros
        metric_scores = {metric: 0 for metric in self.available_metrics}  # Inicializar puntuaciones de métricas
        for other_model in self.results:
            if model != other_model:
                for metric in self.available_metrics:
                    if self.results[model][metric] > self.results[other_model][metric]:
                        metric_scores[metric] += 1
        # Calcular la puntuación total del modelo
        metric_score = sum(metric_scores.values())
        return metric_score
    
   
    def _sum_metrics(self,model):
        """ Metodo para sumar score"""
        total_score =   sum(self.results[model].values())
        return total_score
    
    def _prom_metrics(self,model):
        """ Metodo para sumar score"""
        values = self.results[model].values()
        total_score =   sum(values)/len(values)
        return total_score

    def evaluated_best_model(self):
        if self.type == "count":
            for model in self.results:
                metric_score = self._count_best_metrics(model)
                if metric_score > self.best_metric_score:
                    self.best_model = model
                    self.best_metric_score = metric_score
        
        if self.type == "sum":
            # Evaluar los modelos y seleccionar el mejor
            for model in self.results:
                total_score = self._sum_metrics(model)
                if self.best_model == None or total_score > self.best_total_score:
                    self.best_model = model
                    self.best_total_score = total_score
            
        return self.best_model
