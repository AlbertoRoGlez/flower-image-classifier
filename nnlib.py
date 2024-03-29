import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict

architectures = {"vgg16": 25088,
                "densenet121": 1024,
                "alexnet": 9216 }

def load_data(where  = "./flowers" ):
    '''
    Arguments:
    - where (optional, default: "./flowers"): Specifies the root directory where the data is located. It is a string representing the path to the root directory containing subdirectories for training, validation, and testing datasets.
    
    Returns:
    - dataloaders: A dictionary containing PyTorch data loaders for the training, validation, and testing datasets. Keys are "train", "valid", and "test", and values are PyTorch data loaders.
    - image_datasets: A dictionary containing PyTorch datasets for the training, validation, and testing datasets. Keys are "train", "valid", and "test", and values are PyTorch datasets created using datasets.ImageFolder with the specified data transformations.
    '''

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {"train": transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   "valid": transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   "test": transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                   }

    directories = {"train": train_dir, 
                "valid": valid_dir, 
                "test" : test_dir}

    image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x])
                    for x in ["train", "valid", "test"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ["train", "valid", "test"]} 

    return dataloaders, image_datasets

def nn_builder(architecture= "vgg16", dropout=0.2, hidden_layer= 4096, learning_rate= 0.001, gpu=True):
    '''
    Arguments:
    - architecture (optional, default: "vgg16"): Specifies the neural network architecture to be used. Supported options are "vgg16", "densenet121", and "alexnet".
    - dropout (optional, default: 0.2): Specifies the dropout probability for the dropout layer in the classifier.
    - hidden_layer (optional, default: 4096): Specifies the number of nodes in the hidden layer of the classifier.
    - learning_rate (optional, default: 0.001): Specifies the learning rate for the Adam optimizer.
    - gpu (optional, default: True): If True and GPU is available, the model will be moved to the GPU.
    
    Returns:
    - model: The neural network model with the specified architecture and a custom classifier.
    - criterian: The loss function used for training the model. It is set to Negative Log Likelihood Loss (nn.NLLLoss()).
    - optimizer: The optimizer used for updating the model weights during training. It is set to Adam optimizer with the specified learning rate.
    '''

    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)        
    elif architecture == "densenet121":
        model = models.densenet121(pretrained=True)
    elif architecture == "alexnet":
        model = models.alexnet(pretrained = True)
    else:
        print(f"{architecture} is not a supported model. Available models: vgg16, densenet121, or alexnet.")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([("inputs", nn.Linear(architectures[architecture], hidden_layer)),
                                            ("relu", nn.ReLU()),
                                            ("dropout",nn.Dropout(dropout)),
                                            ("hidden_layer", nn.Linear(hidden_layer, 102)),
                                            ("output", nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterian = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if torch.cuda.is_available() and gpu:
        model.to("cuda")

    return model, criterian, optimizer

def train_nn(model, criterian, optimizer, dataloaders, image_datasets, epochs=40, gpu=True):
    '''
    Arguments:
    - model: The neural network model to be trained.
    - criterian: The loss function used for training.
    - optimizer: The optimizer used for updating the model weights during training.
    - dataloaders: A dictionary containing PyTorch data loaders for the training, validation, and testing datasets. Keys are "train", "valid", and "test".
    - image_datasets: A dictionary containing PyTorch datasets for the training, validation, and testing datasets. Keys are "train", "valid", and "test".
    - epochs (optional, default: 40): Specifies the number of training epochs.
    - gpu (optional, default: True): If True and GPU is available, the model will be trained on the GPU.
    
    Returns:
    - None
    '''
    
    if torch.cuda.is_available() and gpu:
        model.to("cuda")
    
    print("============== NN Training has started. ==============")
    
    for e in range(epochs):

        for dataset in ["train", "valid"]:
            if dataset == "train":
                model.train()  
            else:
                model.eval()   
            
            running_loss = 0.0
            running_accuracy = 0
            
            for inputs, labels in dataloaders[dataset]:
                
                if torch.cuda.is_available() and gpu:
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(dataset == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterian(outputs, labels)

                    # Backward 
                    if dataset == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_accuracy += torch.sum(preds == labels.data)
            
            dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}
            epoch_loss = running_loss / dataset_sizes[dataset]
            epoch_accuracy = running_accuracy.double() / dataset_sizes[dataset]
            
            print(f"Epoch: {e+1}/{epochs}... {dataset} Loss: {epoch_loss:.4f}    Accuracy: {epoch_accuracy:.4f}")

    print("============== NN Training has finished. ==============")

def save_checkpoint(model, image_datasets, filepath="trained_model.pth", architecture ="vgg16", hidden_layer=4096, dropout=0.2, learning_rate=0.001, epochs=40):
    '''
    Arguments:
    - model: The trained neural network model to be saved.
    - image_datasets: A dictionary containing PyTorch datasets for the training, validation, and testing datasets. Keys are "train", "valid", and "test".
    - filepath (optional, default: "trained_model.pth"): Specifies the file path where the checkpoint will be saved.
    - architecture (optional, default: "vgg16"): Specifies the neural network architecture used in the model.
    - hidden_layer (optional, default: 4096): Specifies the number of nodes in the hidden layer of the classifier.
    - dropout (optional, default: 0.2): Specifies the dropout probability for the dropout layer in the classifier.
    - learning_rate (optional, default: 0.001): Specifies the learning rate used during training.
    - epochs (optional, default: 40): Specifies the number of training epochs.
    
    Returns:
    - None
    '''
    
    model.class_to_idx = image_datasets["train"].class_to_idx
    model.cpu

    checkpoint = {"architecture": architecture,
                "hidden_layer": hidden_layer,
                "dropout": dropout,
                "lr": learning_rate,
                "epochs": epochs,
                "class_to_idx":model.class_to_idx,
                "state_dict":model.state_dict()}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath="trained_model.pth"):
    '''
    Arguments:
    - filepath (optional, default: "trained_model.pth"): Specifies the file path from which the checkpoint will be loaded.
    
    Returns:
    - model: The neural network model loaded from the checkpoint file.
    '''
    checkpoint = torch.load(filepath)
    architecture = checkpoint["architecture"]
    hidden_layer = checkpoint["hidden_layer"]
    dropout = checkpoint["dropout"]
    learning_rate = checkpoint["lr"]
    model, _, _ = nn_builder(architecture, dropout, hidden_layer, learning_rate)
    # Load the class_to_idx mapping and model state_dict
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])

    return model

def process_image(image_path):
    '''
    Arguments:
    - image_path: The file path to the image that needs to be processed.
    
    Returns:
    - img_tensor: A PyTorch tensor representing the processed image.
    '''

    img_pil  = Image.open(image_path)
    adjustments  = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img_tensor  = adjustments(img_pil)

    return img_tensor

def predict(image_path, model, topk=5, gpu=True):
    '''
    Arguments:
    - image_path: The file path to the image for which predictions are to be made.
    - model: The trained neural network model used for prediction.
    - topk (optional, default: 5): The number of top predictions to return.
    - gpu (optional, default: True): If True and GPU is available, the model will be used for prediction on the GPU.
    
    Returns:
    - probability: A tuple containing two tensors:
    The first tensor contains the top probabilities for each class.
    The second tensor contains the indices of the top classes.
    '''

    if torch.cuda.is_available() and gpu:
        model.to("cuda")
    
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu:
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
