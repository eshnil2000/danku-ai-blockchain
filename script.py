
======OPTION #2 WORKING OPTION=======
git clone https://github.com/ethereum/web3.py.git && cd web3.py
docker-compose up -d
docker-compose exec sandbox bash

inside container: 
pip3.6 install --upgrade tensorflow
git clone https://github.com/eshnil2000/danku-ai-blockchain.git && cd danku-ai-blockchain
export PYTHONPATH=/code/danku-ai-blockchain
pip3.6 install -r requirements.txt

useful tools
pip install pip-check
pip3.6 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip3.6 install -U

======================================
in solidity ide:
compile, deploy contract Danku_demo_final.sol, get address: 
cd /code/danku-ai-blockchain
in python3.6 console: 
#Automated###
exec(open("script.py").read())
###Manual ####
Danku_demo contract abi
abi=[
	{
		"constant": "true",
		"inputs": [],
		"name": "init1_block_height",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [],
		"name": "init2",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [
			{
				"name": "submission_index",
				"type": "uint256"
			}
		],
		"name": "evaluate_model",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "submission_index",
				"type": "uint256"
			},
			{
				"name": "data",
				"type": "int256[3][]"
			}
		],
		"name": "model_accuracy",
		"outputs": [
			{
				"name": "",
				"type": "int256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "get_training_index",
		"outputs": [
			{
				"name": "",
				"type": "uint256[8]"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "evaluation_stage_block_size",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "",
				"type": "uint256"
			},
			{
				"name": "",
				"type": "uint256"
			}
		],
		"name": "test_data",
		"outputs": [
			{
				"name": "",
				"type": "int256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [
			{
				"name": "_hashed_data_groups",
				"type": "bytes32[10]"
			},
			{
				"name": "accuracy_criteria",
				"type": "int256"
			},
			{
				"name": "organizer_refund_address",
				"type": "address"
			}
		],
		"name": "init1",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "get_testing_index",
		"outputs": [
			{
				"name": "",
				"type": "uint256[2]"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [
			{
				"name": "_test_data_groups",
				"type": "int256[]"
			},
			{
				"name": "_test_data_group_nonces",
				"type": "int256"
			}
		],
		"name": "reveal_test_data",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "paymentAddress",
				"type": "address"
			},
			{
				"name": "num_neurons_input_layer",
				"type": "uint256"
			},
			{
				"name": "num_neurons_output_layer",
				"type": "uint256"
			},
			{
				"name": "num_neurons_hidden_layer",
				"type": "uint256[]"
			},
			{
				"name": "weights",
				"type": "int256[]"
			},
			{
				"name": "biases",
				"type": "int256[]"
			}
		],
		"name": "get_submission_id",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "best_submission_index",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "use_test_data",
		"outputs": [
			{
				"name": "",
				"type": "bool"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [
			{
				"name": "_train_data_groups",
				"type": "int256[]"
			},
			{
				"name": "_train_data_group_nonces",
				"type": "int256"
			}
		],
		"name": "init3",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "l_nn",
				"type": "uint256[]"
			},
			{
				"name": "input_layer",
				"type": "int256[]"
			},
			{
				"name": "hidden_layers",
				"type": "int256[]"
			},
			{
				"name": "output_layer",
				"type": "int256[]"
			},
			{
				"name": "weights",
				"type": "int256[]"
			},
			{
				"name": "biases",
				"type": "int256[]"
			}
		],
		"name": "forward_pass2",
		"outputs": [
			{
				"name": "",
				"type": "int256[]"
			}
		],
		"payable": "false",
		"stateMutability": "pure",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "organizer",
		"outputs": [
			{
				"name": "",
				"type": "address"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "init_level",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"name": "testing_partition",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "get_train_data_length",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "best_submission_accuracy",
		"outputs": [
			{
				"name": "",
				"type": "int256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [],
		"name": "finalize_contract",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "contract_terminated",
		"outputs": [
			{
				"name": "",
				"type": "bool"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "init3_block_height",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "get_submission_queue_length",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [
			{
				"name": "payment_address",
				"type": "address"
			},
			{
				"name": "num_neurons_input_layer",
				"type": "uint256"
			},
			{
				"name": "num_neurons_output_layer",
				"type": "uint256"
			},
			{
				"name": "num_neurons_hidden_layer",
				"type": "uint256[]"
			},
			{
				"name": "weights",
				"type": "int256[]"
			},
			{
				"name": "biases",
				"type": "int256[]"
			}
		],
		"name": "submit_model",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"name": "training_partition",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "reveal_test_data_groups_block_size",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "false",
		"inputs": [],
		"name": "cancel_contract",
		"outputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "model_accuracy_criteria",
		"outputs": [
			{
				"name": "",
				"type": "int256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [
			{
				"name": "",
				"type": "uint256"
			},
			{
				"name": "",
				"type": "uint256"
			}
		],
		"name": "train_data",
		"outputs": [
			{
				"name": "",
				"type": "int256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "get_test_data_length",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": "true",
		"inputs": [],
		"name": "submission_stage_block_size",
		"outputs": [
			{
				"name": "",
				"type": "uint256"
			}
		],
		"payable": "false",
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"payable": "false",
		"stateMutability": "nonpayable",
		"type": "constructor"
	},
	{
		"payable": "true",
		"stateMutability": "payable",
		"type": "fallback"
	}
]

## import required modules
from dutils.dataset import DemoDataset, SampleCircleDataset, SampleAcrossCornerDataset
from dutils.neural_network import NeuralNetwork
import dutils.debug as dbg
from secrets import randbelow
import web3
from web3 import Web3

## Initialize data 
scd = DemoDataset(training_percentage=0.8,partition_size=5)
## Initialize web3 module, increase httpprovider timeout
web3 = Web3(Web3.HTTPProvider("http://nik.chainapp.live:9545",request_kwargs={'timeout':60}))

#set parameters
accuracy_criteria = 5000
offer_account = web3.eth.accounts[1]
solver_account = web3.eth.accounts[2]

## Set up data, generate nonce, add nonce to each group of data partition
scd.generate_nonce()
scd.sha_all_data_groups()

# set default account
web3.eth.defaultAccount=web3.eth.accounts[1]

# Change this address to your contract address
contract_address='0x23bae04a260661d5afbf467f8a48c2fce58fc13e'
myContract=web3.eth.contract(address=web3.toChecksumAddress(contract_address),abi=abi)

## check the dates/blocks when solution can be submitted, evalauted, revealed
submission_t=myContract.functions.submission_stage_block_size().call()

evaluation_t=myContract.functions.evaluation_stage_block_size().call()

test_reveal_t=myContract.functions.reveal_test_data_groups_block_size().call()


## Initialize contract with the hashed data, accuracy criteria, set 
init1_tx=myContract.functions.init1(scd.hashed_data_group,accuracy_criteria,web3.toChecksumAddress(offer_account)).transact()

# estimate gas if required
myContract.functions.init1(scd.hashed_data_group,accuracy_criteria,web3.toChecksumAddress(offer_account)).estimateGas()

init1_block_number = web3.eth.blockNumber

# Fund the contract, set the reward
web3.eth.sendTransaction({'from': web3.eth.accounts[0], 'to':Web3.toChecksumAddress(contract_address),'value': web3.toWei(0.0001, "ether")})

# Initialize step 2 , randomly select indices of partitions for training & testing
init2_block_number = web3.eth.blockNumber
init2_tx = myContract.functions.init2().transact()

training_partition = list(map(lambda x:myContract.functions.training_partition(x).call(),range(scd.num_train_data_groups)))
testing_partition = list(map(lambda x:myContract.functions.testing_partition(x).call(),range(scd.num_test_data_groups)))
scd.partition_dataset(training_partition, testing_partition)

# append training nonces
training_nonces = []
training_data = []
for i in training_partition:
	training_nonces.append(scd.nonce[i])

# Pack data, read for submission to blockchain
# Initial data is : [(x1,y1,answer1), (x2,y2,answer2)...]
# Packed data: [x1,y1,answer1,x2,y2,answer2,...]
train_data = scd.pack_data(scd.train_data)
test_data = scd.pack_data(scd.test_data)
init3_tx = []

for i in range(len(training_partition)):
	start = i*scd.dps*scd.partition_size
	end = start + scd.dps*scd.partition_size
	iter_tx = myContract.functions.init3(train_data[start:end], scd.train_nonce[i]).transact()
	init3_tx.append(iter_tx)

# get training data (this is called by participant)
contract_train_data_length=myContract.functions.get_train_data_length().call()
contract_train_data = []

# get training data , unpack.  DONE BY PARTICPANT 
for i in range(contract_train_data_length):
	for j in range(scd.dps):
		contract_train_data.append(myContract.functions.train_data(i,j).call())

contract_train_data = scd.unpack_data(contract_train_data)

# DONE BY PARTICIPANT. TRAIN OFFLINE WITH DATA
il_nn = 2
hl_nn = [4,4]
ol_nn = 2
nn = NeuralNetwork(il_nn, hl_nn, ol_nn, 0.1, 100, 5, 100)

#Convert to 1 hot encoding
contract_train_data = nn.binary_2_one_hot(contract_train_data)

nn.load_train_data(contract_train_data)
nn.init_network()
nn.train()
trained_weights = nn.weights
trained_biases = nn.bias
#### 

### prepare the weights and biases for submission
packed_trained_weights = nn.pack_weights(trained_weights)

packed_trained_biases = nn.pack_biases(trained_biases)
def scale_packed_data(data, scale):
	# Scale data and convert it to an integer
	return list(map(lambda x: int(x*scale), data))

w_scale = 1000 # Scale up weights by 1000x
b_scale = 1000 # Scale up biases by 1000x
int_packed_trained_weights = scale_packed_data(packed_trained_weights,w_scale)
int_packed_trained_biases = scale_packed_data(packed_trained_biases,\
		b_scale)

### Submit solution
submit_tx = myContract.functions.submit_model(solver_account, il_nn, ol_nn, hl_nn,int_packed_trained_weights, int_packed_trained_biases).transact()

my_submission_id=myContract.functions.get_submission_id(solver_account, il_nn,ol_nn, hl_nn, int_packed_trained_weights, int_packed_trained_biases).call()

### reveal test data
reveal_tx = []
for i in range(len(testing_partition)):
	start = i*scd.dps*scd.partition_size
	end = start + scd.dps*scd.partition_size
	iter_tx = myContract.functions.reveal_test_data(test_data[start:end], scd.test_nonce[i]).transact()
	reveal_tx.append(iter_tx)

## after the required lock up period is over, evaluate the model, payout to best submission
eval_tx = myContract.functions.evaluate_model(my_submission_id).transact()
final_tx = myContract.functions.finalize_contract().transact()
best_submission_accuracy=myContract.functions.best_submission_accuracy()
best_submission_accuracy.transact()
best_submission_accuracy=myContract.functions.best_submission_accuracy().transact()
best_submission_index=myContract.functions.best_submission_index().call()

####Useful Pandas commands
Assume df2 is dataframe pandas
Assume column names are:
cols_to_use2 = ["fips_fixed", "clinton", "trump", "state", "jurisdiction"]
import pandas as pd

df2 = pd.read_csv("data/election_results_with_demographics.csv", usecols=cols_to_use2)

print row 0
df2.loc[0]

print all columns
df2.columns.values

shape
df2.shape
df2.index

###Python super
class Rectangle:
	def __init__(self, length, width):
		self.length = length
		self.width = width

	def area(self):
		return self.length * self.width

	def perimeter(self):
		return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
	def __init__(self, length):
		super().__init__(length, length)
Here, you’ve used super() to call the __init__() of the Rectangle class, allowing you to use it in the Square class without repeating code. Below, the core functionality remains after making changes:

######


