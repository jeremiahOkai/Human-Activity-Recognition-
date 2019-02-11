import matplotlib.pyplot as plt

class Image(object): 
    def __init__(self, history, model_name): 
        self.history = history
        self.model_name = model_name
        
    def create_image(self): 
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('/home/s1931628/zeropdf/' + self.model_name + '_accuracy.pdf')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('/home/s1931628/zeropdf/' + self.model_name + '_loss.pdf')
        plt.show()