from TrainModel import TrainModel
from GiveResult import GiveResult

def main():
    string = input("Filename (empty for training): ")
    if string == "":
        TrainModel.trainModel() # call trainModel()
    else:
        GiveResult.giveResult(string) # with file open: give results
    

if __name__ == "__main__":
    main()