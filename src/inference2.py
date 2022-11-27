# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import json
# from model import Net
# from decode_base64 import decode_base64
# from words import get_label_name, get_most_similar_word
# # import torchvision.models as models



# def infer(model,image,top_n=5):
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         output = F.softmax(output,dim=1)
#         values, indices = torch.topk(output, top_n)
#         # _, pred = torch.max(output, 1)
#         return values, indices


# if __name__ == '__main__':

#     # print(models.resnet18())
#     # print( Net(rn="resnet18"))
#     first_letter = "ね"

#     net = Net(rn="resnet18")

#     print("")
#     print("loading model...")
#     # net.load_state_dict(torch.load("./weights/RN50_ver2_|_100_per_class_|_50_epochs.pth"))
#     net.load_state_dict(torch.load("./weights/RN18_ver1_|_1000_per_class_|_3_epochs.pth"))
#     print("done.")
    

#     # base64_string = json.load(open("./src/base64.json"))["racket"]#["base64_text"]
#     base64_string = json.load(open("./src/base64.json"))["cat"]#["base64_text"]

#     image_tensor = decode_base64(base64_string,28)
#     print(image_tensor.shape)
#     # print(net.resnet.conv1)
#     # print(list(net.parameters())[0].dtype)
#     # print(image_tensor.dtype)

#     print("")
#     print("running inference...")
#     values, indices = infer(net, image_tensor.unsqueeze(0).unsqueeze(0), top_n=5)
#     print("done.")

#     for value, index in zip(values[0], indices[0]):
#         print(f"{get_label_name(index.item())}: {value.item()}")

#     # top_indice = indices[0]
#     # top_word = get_label_name(top_indice[0].item())
    
#     # print("")
#     # print(f"top word: {top_word}")
#     # print("")

#     # print("getting most similar word...")
#     # print("")
#     # if top_word[0] == first_letter:
#     #     most_similar_word = top_word
#     #     print(f"most similar word is: {most_similar_word}")
#     # else:
#     #     most_similar_word = get_most_similar_word(first_letter,top_word,nearest_n=15)
#     #     print(f"most similar word is: {most_similar_word}")



import torch
import torch.optim as optim
from dataset import QuickDrawDataset, get_loader
from train import train_loop
from eval import eval_loop
from decode_base64 import plot_image_tensor
from model import Net,CNN
import torch.nn as nn
from tqdm import tqdm
import wandb
from wandb import AlertLevel


if __name__ == '__main__':
    model_name = "RN18_ver1_|_1000_per_class_|_3_epochs"
    which_resnet = "resnet18"
    # model_name = "CNN_ver1_|_1000_per_class_|_3_epochs"
    data_dir = './data'
    max_examples_per_class = 10#15000 #15000
    train_val_split_pct = .1
    lr = 0.01
    num_epochs = 5
    batch_size = 128
    shuffle = True
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ds = QuickDrawDataset(
        root=data_dir,
        max_items_per_class=max_examples_per_class,
        class_limit=None,
        is_download=False,
    )
    
    train_ds, val_ds = ds.split(train_val_split_pct)
    
    train_loader = get_loader(train_ds, batch_size, shuffle, num_workers)
    val_loader = get_loader(val_ds, batch_size, shuffle, num_workers)

    # sample_images, sample_labels = next(iter(train_loader))
    # print(sample_images.shape)
    
    # net = Net(rn = which_resnet).to(device)
    # # net = CNN().to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=-1)
    
    # print single example 
    
    example = next(iter(train_loader))
    # print(example)
    print(example[0][0][0].shape)
    print(example[0][0][0].sum())
    # print(example[0][0][0].mean())

    # plot_image_tensor(example[0][0][0],"./src/example.png")

    import json
    from decode_base64 import decode_base64

    base64_string = json.load(open("./src/base64.json"))["cat"]#["base64_text"]
    # base64_string = json.load(open("./src/base64.json"))["cat"]#["base64_text"]

    image_tensor = decode_base64(base64_string,28)
    print(image_tensor.sum())
    print(image_tensor.shape)

    concat_tensor = torch.cat((example[0][0][0],image_tensor.squeeze(0)),dim=0)
    print(concat_tensor.shape)
    print(concat_tensor)


    plot_image_tensor(concat_tensor,"./src/concat_tensor.png")

    # wandb_config = dict(
    #     project="orf_eshiritori",
    #     group="CNN",
    #     name="CNN ver1 | 1000 per class | 3 epochs"
    # )
    # wandb_config = dict(
    #     project="orf_eshiritori",
    #     group="RN",
    #     name="RN18 ver1 | 1000 per class | 3 epochs"
    # )

    # with wandb.init(job_type="train",**wandb_config):

    #     print("starting training...")
    #     best_val_acc = 0

    #     for epoch in tqdm(range(1, num_epochs)):

    #         loss = train_loop(net, train_loader, device, criterion, optimizer)
    #         acc = eval_loop(net, val_loader, device, criterion)

    #         print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")
    #         wandb.log({"epoch":epoch,"loss":loss,"acc":acc,"lr":optimizer.param_groups[0]['lr']})

    #         if acc > best_val_acc:
    #             best_val_acc = acc
    #             print("saving model...")
    #             torch.save(net.state_dict(), f"weights/{model_name}_best.pth")
    #             print("done.")

    #         scheduler.step()

    #     print("done.")

    #     print("saving model...")
    #     # save the model
    #     torch.save(net.state_dict(), f"weights/{model_name}.pth")
    #     print("done.")

    # wandb.alert(
    # title="ORF 絵しりとり",
    # text="<@U013HNPE0GG> 学習終了しました",
    # level=AlertLevel.INFO
    # )