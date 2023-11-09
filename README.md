# Extra Credit Project: 5 Points
# Image Generation with GAN

#### Due Date
* Tuesday Dec. 12, 2023 (23:59)

#### Total Points 
* 5

## Goal
In this assignment you will be asked to implement a Generative Adversarial Networks (GAN) with [MNIST data set](http://yann.lecun.com/exdb/mnist/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/tutorials/). 

<img src="https://github.com/XinZhang525/CS549ML/blob/main/goal.png" width="80%">

## Project Guidelines

#### Data set

MNIST is a dataset composed of handwrite numbers and their labels. Each MNIST image is a 28\*28 grey-scale image, which is labeled with an integer value from 0 and 9, corresponding to the actual value in the image. MNIST is provided in Pytorch as 28\*28 matrices containing numbers ranging from 0 to 255. There are 60000 images and labels in the training data set and 10000 images and labels in the test data set. Since this project is an unsupervised learning project, you can only use the 60000 images for your training. 

#### Installing Software and Dependencies 

* [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
* [Create virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Install packages (e.g. pip install torch)

#### Building and Compiling Generator and Discriminator

In Pytorch, you can try different layers, such as “Conv2D”, different activation functions, such as “tanh”, “leakyRelu”. You can apply different optimizers, such as stochastic gradient descent or Adam, and different loss functions. The following is the sample code of how to build the model.


```python
# Create a Generator class.
class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Generator.
netG = Generator(*args)

# Create a Discriminator class.
class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        # Define your network architecture.

    def forward(self, x):
        # Define your network data flow. 
        return output
# Create a Discriminator.
netD = Discriminator(*args)

# Setup Generator optimizer.
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Setup Discriminator optimizer.
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999))

# Define loss function. 
criterion = torch.nn.BCELoss()
```

#### Training GAN

You have the option of changing how many epochs to train your model for and how large your batch size is. The following is the sample code of how to train GAN. You can add self-defined parameters such as #epoch, learning rate scheduler to the train function.



```python
# Training
def train():
    for _ in range(batchCount):  
	
        # Create a batch by drawing random index numbers from the training set
       
        # Create noise vectors for the generator
        
        # Generate the images from the noise

        # Create labels

        # Train discriminator on generated images

        # Train generator

```

#### Saving Generator

Please use the following code to save the model and weights of your generator.



```python
# save model with Pytorch
torch.save(netG.state_dict(), 'PATH_TO_SAVED_GENERATOR')
torch.save(netD.state_dict(), 'PATH_TO_SAVED_DISCRIMINATOR')
```

#### Plotting

Please use the following code to plot the generated images. As for the loss plot of your generator and discriminator during the training, you can plot with your own style. 


```python
# Generate images
np.random.seed(504)
h = w = 28
num_gen = 25

z = np.random.normal(size=[num_gen, z_dim])
generated_images = netG(z)

# plot of generation
n = np.sqrt(num_gen).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(I_generated, cmap='gray')
plt.show()
```

## Deliverables

Please compress all the below files into a zipped file and submit the zip file (firstName_lastName_GAN.zip) to Canvas. 

#### Python code
* Include model creation, model training, plotting code.

#### Generator Model
* Turn in your best generator saved as “generator.pt” and the weights of your generator saved as “generator_weights.pt”.
