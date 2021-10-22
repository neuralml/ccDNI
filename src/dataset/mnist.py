import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as trans
import torchvision.datasets as dataset

import numpy as np

def load_raw(data_path, input_size, download=False, transform=None, target_transform=None):
    transform = transform or trans.Compose([
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(-1, input_size))
    ])

    train_data = dataset.MNIST(
        root=data_path, train=True, transform=transform, target_transform=target_transform, download=download)
    test_data = dataset.MNIST(
        root=data_path, train=False, transform=transform, target_transform=target_transform, download=download)

    return train_data, test_data


def load_mnist(data_path, input_size, batch_size, val_split=0.2,
        shuffle=True, download=False, transform=None, target_transform=None, numbers=None
    ):
    train_raw, test_raw = load_raw(data_path, input_size, download=download,
                                   transform=transform, target_transform=target_transform)
    
    
    if numbers is not None:
        train_raw = filter_mnist(train_raw, numbers)
        test_raw = filter_mnist(test_raw, numbers)
    
    # Split train data into training and validation sets
    N = len(train_raw)
    val_size = int(N * val_split)
    train_raw, validation_raw = random_split(
        train_raw, [N - val_size, val_size])
    
    train_data = DataLoader(
        train_raw, batch_size=batch_size, shuffle=shuffle)
    validation_data = DataLoader(
        validation_raw, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(
        test_raw, batch_size=batch_size, shuffle=False)

    return train_data, validation_data, test_data


def draw_2ddigits(seq_len, digits):
    if digits is None:
        digits = np.arange(10)
    n = len(digits)
    X = np.zeros((n, seq_len, 2))
    
    for i in range(n):
        methodname = "draw" + str(digits[i])
        drawdigit = globals()[methodname]
        digit = drawdigit(seq_len)
        X[i] = digit
    
    return torch.from_numpy(X).float()

def draw_2dlines(nlines, seq_len, fixed_mag=True):
    X = np.zeros((nlines, seq_len, 2))
   
    if fixed_mag:
        end_points = get_circle_points(nlines)
    else:
        end_points = get_2dpoints(nlines) #n unique end points for the line
        end_points = end_points * 10 

    for i in range(nlines):
        #end_point = torch.rand(2)
        line = draw_2dline(seq_len, end_points[i])
        X[i] = line
    
    return torch.from_numpy(X).float()
    
    
    
def draw_2dline(seq_len, end_point, start_point = None):
    #pick a random start point and end point on the square, 
    #and make sequence of points between them
    if start_point is None: 
        start_point = np.zeros(2) #each line starts at the origin
    
    seqx = np.linspace(start_point[0], end_point[0], num=seq_len)
    seqy = np.linspace(start_point[1], end_point[1], num=seq_len)
    
    line = np.vstack((seqx, seqy)).T
    
    return line
    
def get_2dpoints(npoints):
    if npoints > 16:
        print("Unexpectedly high number of points being asked for (>16). Method 'get_2dpoints()' should be appropriately updated")
    x = np.linspace(0, 1, 4) #4 is sqrt(16)   
    y = np.linspace(0, 1, 4)
    xx, yy = np.meshgrid(x, y)
    
    allpoints = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), 1)
    inds = np.random.randint(0,16,npoints)
    inds = np.random.choice(16, npoints, replace=False)
    
    points = allpoints[inds]
    
    return torch.from_numpy(points).float()


def get_circle_points(npoints, mag=10):
    angles = np.linspace(0, 2*np.pi, npoints, endpoint=False)
    coords = np.vstack((np.sin(angles), np.cos(angles))).T
    coords = mag * coords
    return torch.from_numpy(coords).float()


def filter_mnist(mnist_dataset, numbers):
    inds = np.isin(mnist_dataset.targets, numbers)
    
    mnist_dataset.data = mnist_dataset.data[inds]
    mnist_dataset.targets = mnist_dataset.targets[inds]
    
    return mnist_dataset


#individual methods for each number...probably easier
#assume same 3x3 grid space for all numbers
def fit_points(seq_len, points):
    npairs = points.shape[0] - 1
    assert seq_len >= len(points), "Something went wrong"
    
    div = seq_len // npairs
    r = seq_len % npairs
    
    
    lengths = np.ones(npairs) * div
    
    if r > 0:
        lengths[:r-1] += 1 #black magic
    else:
        lengths[-1] -= 1
    

    lines = []    
    for i in range(npairs): #last pair is weird, deal with separately
        start_point = points[i]
        end_point = points[i + 1]
        line = draw_2dline(lengths[i] + 1, end_point, start_point)[:-1] #let next point be represented in the next pair
        lines.append(line)
    
    lines.append(points[-1].reshape(1, 2))
    
    lines = np.vstack(lines)
    
    assert lines.shape == (seq_len, 2), "Something went wrong"
    
    return lines    

#start from [0,1] and go counter clockwise
def draw0(seq_len):  
    #npoints = len(x)   
    x = np.array([0, 0, 0, 0.5, 1, 1, 1, 0.5, 0])
    y = np.array([1, 0.5, 0, 0, 0, 0.5, 1, 1, 1])
    points = np.vstack((x,y)).T
    
    curve = fit_points(seq_len, points)
    return curve
        
def draw1(seq_len):
    x = np.array([0.25, 0.5, 0.5, 0.5])
    y = np.array([0.75, 1, 0.5, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve

def draw2(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0, 0.5, 1])
    y = np.array([1, 1, 1, 0.5, 0, 0,0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve   

def draw3(seq_len):
    x = np.array([0, 0.5, 1, 1, 0.5, 1, 1, 0.5, 0])
    y = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0, 0,0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve    

def draw4(seq_len):
    #x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 1])
    #y = np.array([1, 0.5, 0, 0.5, 1, 0.5, 0.5, 0.5])
    x = np.array([0, 0, 0.5, 1, 0.5, 0.5, 0.5, 0.5])
    y = np.array([1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve       

def draw5(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0, 0, 0.5, 1, 1, 0.5, 0])
    y = np.array([1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve 

def draw6(seq_len):
    x = np.array([0, 0, 0, 0.5, 1, 1, 0.5, 0])
    y = np.array([1, 0.5, 0, 0, 0, 0.5, 0.5, 0.5])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve

def draw7(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0])
    y = np.array([1, 1, 1, 0.5, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve

def draw8(seq_len):
    x = np.array([0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0])
    y = np.array([1, 0.5, 0, 0, 0, 0.5, 1, 1, 1])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve  

def draw9(seq_len):
    x = np.array([0, 0, 0.5, 1, 1, 0.5, 0, 0.5, 1, 1, 1])
    y = np.array([1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 0.5, 0])
    points = np.vstack((x,y)).T
    curve = fit_points(seq_len, points)
    return curve     
  
