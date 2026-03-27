📌 Loss Function Behaviour in Neural Network Optimization



* This project investigates the effect of different loss functions on gradient behaviour and training stability in neural networks.





🚀 Objective



To compare:



* Binary Cross-Entropy (BCE)
* Mean Squared Error (MSE)



under saturated neuron conditions.





🧪 Key Idea



Neural networks suffer from vanishing gradients, especially when activation functions saturate.



This experiment:



* Forces saturation via high weight initialization
* Tracks gradient norms during training
* Compares how BCE and MSE behave





⚙️ Technologies Used



* Python
* TensorFlow
* NumPy
* Matplotlib
* Scikit-learn





📊 Experiment Setup



* Dataset: make\_moons
* Model: 2-layer neural network (tanh + sigmoid)
* Optimizer: SGD
* Metric tracked:
* Loss convergence
* Gradient norm (last layer)





📈 Results



* Loss Function	Behaviour
* BCE	Maintains stronger gradients
* MSE	Suffers from vanishing gradients





🔍 Key Insight



Binary Cross-Entropy provides better gradient flow compared to Mean Squared Error in classification tasks, especially under saturation.





▶️ How to Run



pip install -r requirements.txt

python main.py





📁 Output



Loss convergence plot

Gradient magnitude plot (log scale)





📌 Research Contribution



This project demonstrates experimentally:



Why BCE is preferred over MSE for classification

The role of loss functions in optimization landscapes

