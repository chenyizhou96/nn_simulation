import sys, getopt

class TrainConfig:
  def __init__(self, argv):
    self.model_list = ['LINEAR_ELASTICITY', 'COROTATED']
    self.activation_list = ['RELU', 'SIGMOID', 'TANH']
    self.optimizer_list = ['SGD', 'ADAM']
    self.model_number = 0
    self.activation = 0
    self.optimizer = 0
    self.learning_rate = 0.001
    self.shuffle = False
    try:
      opts, args = getopt.getopt(argv[1:], '' ,['optimizer=','activation=', 'model=', 'learning_rate=', 'shuffle'])
    except getopt.GetoptError:
      print ('--model=<model_number>, --activation=<activate function>, --optimizer=<optimizer type>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit()
      elif opt == "--model":
        self.model_number = int(arg)
      elif opt == "--activation":
        self.activation = int(arg)
      elif opt == "--optimizer":
        self.optimizer = int(arg)
      elif opt == "--learning_rate":
        self.learning_rate = float(arg)
      elif opt == "--shuffle":
        self.shuffle = True

    print("Argument parsed.")
    print("Optimizer: " + self.optimizer_list[self.optimizer])
    print("Activation Function: " + self.activation_list[self.activation])
    print("Model: " + self.model_list[self.model_number])
    print("Learning rate: " + str(self.learning_rate))
    print("Shuffle: " + str(self.shuffle))


