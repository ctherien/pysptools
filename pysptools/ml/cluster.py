#

# belong to the skl module?
class Cluster(object):
    
    def __init__(self, estimator:None, estimator_param:None):
        self.estimator = estimator
        self.estimator_param = estimator_param
        self.roi = None
        self.model = None
        
    def classify(self, M, ROIs):
        self.roi = ROIs
        if self.estimator_param != None:
            self.model = self.estimator(**self.estimator_param)
        else:
            self.model = self.estimator()
        self.model.fit_rois(M, self.roi)
        self.model.classify(M)
        return self.model.get_class_map()
    
    def plot(self, path, interpolation=None, colorMap='Paired', suffix=None):
        self.model.plot(path, labels=self.roi.get_labels(), interpolation=interpolation, 
                           colorMap=colorMap, suffix=suffix)

    def display(self, interpolation=None, colorMap='Paired', suffix=None):
        self.model.display(labels=self.roi.get_labels(), interpolation=interpolation, 
                           colorMap=colorMap, suffix=suffix)
