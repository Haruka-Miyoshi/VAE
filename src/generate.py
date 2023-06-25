from variationalautoencoder import VAE
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    # 検証データ
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download = True)
    # VAE
    vae=VAE(zd=64, mode=True, model_path='./model/parameter.pth')

    for n in range(10):
        x,y=test_dataset[n]

        x_hat=vae.generated(x)

        x_hat=x_hat.to('cpu').detach()

        plt.imshow(x_hat.view(-1,28), cmap='gray')

        plt.savefig(f'./figs/{y}_{n}.png')
        
if __name__=='__main__':
    main()