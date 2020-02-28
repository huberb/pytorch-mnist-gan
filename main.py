import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gen(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Gen, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(in_shape, 512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, out_shape),
                nn.Tanh()
                ).to(device)

    def forward(self, x):
        return self.model(x)

class Dis(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Dis, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(in_shape, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, out_shape),
                nn.Softmax(1)
                ).to(device)

    def forward(self, x):
        return self.model(x)

def train_dis(gen, dis, images, labels, crit_dis, optim_dis):
    dis.zero_grad()
    # train on real data
    output = dis(images.to(device))
    real_image_labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
    real_images_loss = crit_dis(output, real_image_labels)
    real_images_loss.backward()
    # train on fake data
    generated_image = gen(real_image_labels).detach()
    output = dis(generated_image)
    fake_images_loss = crit_dis(output, torch.zeros(output.shape).to(device))
    fake_images_loss.backward()
    # train dis
    optim_dis.step()
    return real_images_loss.item() + fake_images_loss.item()

def train_gen(gen, dis, images, labels, crit_gen, optim_gen, batch_count):
    gen.zero_grad()
    # run labels through both models
    input_gen = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
    output_gen = gen(input_gen)
    output_dis = dis(output_gen)
    loss = crit_gen(output_dis, input_gen)
    loss.backward()
    # train gen
    optim_gen.step()
    return loss.item()

def save_images(gen, epoch, labels_count):
    target = torch.arange(labels_count, dtype=torch.int64)
    gen_input = torch.nn.functional.one_hot(target, num_classes=labels_count)
    images = gen(gen_input.float().to(device))
    images = images.detach().cpu()
    torchvision.utils.save_image(images.view(gen_input.shape[0], 1, 28, 28),
            f"./results/epoch_{epoch}.png",
            nrow=5)

def train():
    EPOCHS = 100
    BATCH_COUNT = 64
    IMG_SIZE = 784
    LABELS_COUNT = 10
    LR_GEN = 0.0001
    LR_DIS = 0.0001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])

    mnist_trainset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
            )
    train_loader1 = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=BATCH_COUNT)
    train_loader2 = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=BATCH_COUNT)

    gen = Gen(LABELS_COUNT, IMG_SIZE)
    dis = Dis(IMG_SIZE, LABELS_COUNT)

    criterion_gen = nn.MSELoss()
    optim_gen = optim.Adam(gen.parameters(), lr=LR_GEN)

    criterion_dis = nn.MSELoss()
    optim_dis = optim.Adam(dis.parameters(), lr=LR_DIS)

    losses_dis = []
    losses_gen = []

    for epoch in range(EPOCHS):
        sum_dis_loss = 0
        sum_gen_loss = 0
        print(f"\nSTARTING EPOCH #{epoch}")
        now = time.time()
        for i, (images, labels) in enumerate(train_loader1):
            images = images.view(-1, IMG_SIZE)
            dis_loss = train_dis(gen, dis, images, labels, criterion_dis, optim_dis)
            sum_dis_loss += dis_loss
            if i == 100:
                break;
        for i, (images, labels) in enumerate(train_loader2):
            images = images.view(-1, IMG_SIZE)
            gen_loss = train_gen(gen, dis, images, labels, criterion_gen, optim_gen, BATCH_COUNT)
            sum_gen_loss += gen_loss
            if i == 100:
                break;
        losses_dis.append(sum_dis_loss)
        losses_gen.append(sum_gen_loss)
        print(f"\nEPOCH {epoch} FINISHED, LOSS DIS: {sum_dis_loss}, LOSS GEN: {sum_gen_loss}")
        then = time.time()
        print(f"TIME OF EPOCH: {then - now}\n")
        save_images(gen, epoch, LABELS_COUNT)
    plt.plot(list(range(len(losses_dis))), losses_dis)
    plt.plot(list(range(len(losses_gen))), losses_gen)
    plt.savefig("./losses.png")

if __name__ == "__main__":
    train()
