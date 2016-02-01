#!/Users/panix/anaconda/bin/python

print "Tic"

from pybrain.structure import FeedForwardNetwork, SigmoidLayer, LinearLayer
from pybrain.structure import FullConnection, LinearConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

import networkx as nx

import collections
import copy
import sys

#import numpy as np
#import matplotlib.pyplot as plt



# First we need a test network

test_g = nx.MultiDiGraph()

# Target gene
test_g.add_node("RV1026", type="gene")

# Regulatory elements
test_g.add_node("RV0102", type="gene,TF")
test_g.add_node("RV3056", type="gene,TF")
test_g.add_node("RV0912", type="gene,TF")
test_g.add_node("RV0007", type="gene,TF")

test_g.add_node("RV0972", type="not_connected")

# Other element to ignore
test_g.add_node("RV3296", type="other")

# Edges including energy value
test_g.add_edges_from([("RV0102","RV1026",{'weight':1}),("RV3056","RV1026",{'weight':3}),("RV0007","RV1026",{'weight':4}),("RV0912","RV1026",{'weight':2}),("RV3296","RV1026")])

# Write to visualize
nx.write_gml(test_g, "test.gml")

#print test_g.edges()
#print test_g["Gene_1"]["TF_1"]["Energy"]

# Done

# Mock test dataset

training_data = []

exp_dataset_1 = {}
exp_dataset_2 = {}
exp_dataset_3 = {}
exp_dataset_4 = {}

exp_dataset_1["RV0102"] = '2'
exp_dataset_1["RV3056"] = '3'
exp_dataset_1["RV0912"] = '2'
exp_dataset_1["RV0007"] = '4'
exp_dataset_1["RV1026"] = '19'

exp_dataset_2["RV0102"] = '5'
exp_dataset_2["RV3056"] = '3'
exp_dataset_2["RV0912"] = '1'
exp_dataset_2["RV0007"] = '1'
exp_dataset_2["RV1026"] = '16'

exp_dataset_3["RV0102"] = '1'
exp_dataset_3["RV3056"] = '4'
exp_dataset_3["RV0912"] = '2'
exp_dataset_3["RV0007"] = '3'
exp_dataset_3["RV1026"] = '20'

exp_dataset_4["RV0102"] = '2'
exp_dataset_4["RV3056"] = '7'
exp_dataset_4["RV0912"] = '11'
exp_dataset_4["RV0007"] = '4'
exp_dataset_4["RV1026"] = '49'



training_data.append(exp_dataset_1)
training_data.append(exp_dataset_2)
training_data.append(exp_dataset_3)
training_data.append(exp_dataset_4)



bias_training_data = []

exp_dataset_1 = {}
exp_dataset_2 = {}
exp_dataset_3 = {}
exp_dataset_4 = {}
exp_dataset_5 = {}

exp_dataset_1["rv0102"] = '1'
exp_dataset_1["rv3056"] = '12'
exp_dataset_1["rv0912"] = '2'
exp_dataset_1["rv0007"] = '5'
exp_dataset_1["rv1026"] = '10'

exp_dataset_2["rv0102"] = '5'
exp_dataset_2["rv3056"] = '3'
exp_dataset_2["rv0912"] = '4'
exp_dataset_2["rv0007"] = '1'
exp_dataset_2["rv1026"] = '19'

exp_dataset_3["rv0102"] = '1'
exp_dataset_3["rv3056"] = '15'
exp_dataset_3["rv0912"] = '6'
exp_dataset_3["rv0007"] = '3'
exp_dataset_3["rv1026"] = '28'

exp_dataset_4["rv0102"] = '2'
exp_dataset_4["rv3056"] = '6'
exp_dataset_4["rv0912"] = '4'
exp_dataset_4["rv0007"] = '4'
exp_dataset_4["rv1026"] = '19'

exp_dataset_5["rv0102"] = '2'
exp_dataset_5["rv3056"] = '6'
exp_dataset_5["rv0912"] = '10'
exp_dataset_5["rv0007"] = '4'
exp_dataset_5["rv1026"] = '46'



bias_training_data.append(exp_dataset_1)
bias_training_data.append(exp_dataset_2)
bias_training_data.append(exp_dataset_3)
bias_training_data.append(exp_dataset_4)
bias_training_data.append(exp_dataset_5)



class RPropMinusTrainer_Evolved(RPropMinusTrainer):
	def train(self):
		""" Train the network for one epoch """
		from scipy import sqrt
		self.module.resetDerivatives()
		errors = 0
		ponderation = 0
		for seq in self.ds._provideSequences():
			e, p = self._calcDerivs(seq)
			errors += e
			ponderation += p
		if self.verbose:
			print(("epoch {epoch:6d}  total error {error:12.5g}   avg weight  {weight:12.5g}".format(
				epoch=self.epoch,
				error=errors / ponderation,
				weight=sqrt((self.module.params ** 2).mean()))))
		self.module._setParameters(self.descent(self.module.derivs - self.weightdecay * self.module.params))
		self.epoch += 1
		self.totalepochs += 1
		self.total_error = (errors / ponderation)
		return errors / ponderation


class gene_Neuron_Cluster:
	def __init__(self, gml_path):
		self.gml_path = gml_path
		self.gml = nx.read_gml(gml_path, relabel=True)

	def add_xml_file(self, xml_filepath):
		self.xml_filepath = xml_filepath
		self.xml = NetworkReader.readFrom(self.xml_filepath)
		print 'Number of inputs:' , str(len(self.xml.inputbuffer[0]))

	def add_organism(self, organism):
		self.organism = organism

	def predict_output(self, input_dict): 
		input_list = input_dict
		return self.xml.activate(input_list)

	def input_list(self):
			
		#print self.gml.nodes(data=True)

		#for node in self.gml.nodes(data=True):
		#print '\n'
		out_degree_dict = self.gml.out_degree()
		in_degree_dict = self.gml.in_degree()

		input_nodes = []

		for key in out_degree_dict.keys():
			if out_degree_dict[key] > 0 and in_degree_dict[key] == 0:
				input_nodes.append(key)

		#print input_nodes

		# Dealing with the gml network

		gml_dict = {}

		for gml_gene in input_nodes:
			edge_weights = []
			for g_edge in self.gml[gml_gene]:
				edge_weights.append(str(self.gml[gml_gene][g_edge]['weight'])[:6])
			gml_dict[gml_gene] = edge_weights

		#print gml_dict
		#print '\n'

		# Dealing with the xml network

		xml_dict = {}

		for mod in self.xml.modules:
			if mod.name == 'Input_layer':
				for conn in self.xml.connections[mod]:
					#print dir(conn)
					#print conn.params
					#print len(conn.params)
					for cc in range(len(conn.params)):
						#print conn.whichBuffers(cc), conn.params[cc]
						#print conn.whichBuffers(cc)[0], conn.params[cc]

						order_string = str(conn.whichBuffers(cc)[0])

						if order_string in xml_dict.keys():
							xml_dict[order_string].append(str(conn.params[cc])[:6])
						else:
							ent_list = [str(conn.params[cc])[:6]]
							xml_dict[order_string] = ent_list
		#print xml_dict

		mapping_dict = {}

		for in_gene in gml_dict.keys():
			for in_number in xml_dict.keys():
				if collections.Counter(gml_dict[in_gene]) == collections.Counter(xml_dict[in_number]):
					#print in_gene, in_number
					mapping_dict[in_gene] = in_number

		return mapping_dict




def parallel_function(f):
    def easy_parallize(f, sequence):
        """ assumes f takes sequence as input, easy w/ Python's scope """
        # See http://scottsievert.github.io/blog/2014/07/30/simple-python-parallelism/
        from multiprocessing import Pool
        pool = Pool(processes=4) # depends on available cores
        result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
        cleaned = [x for x in result if not x is None] # getting results
        cleaned = asarray(cleaned)
        pool.close() # not optimal! but easy
        pool.join()
        return cleaned
    from functools import partial
    return partial(easy_parallize, f)


def pesos_conexiones(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]

def import_training_data(data_path):
	
	training_data = []

	data_obj = open(data_path, "r")
	if data_path[-3:] == "pcl":
		headder = True
		unformatted_data = {}
		for line in data_obj.readlines():
			if headder == True:
				headder_list_old = line.split("\t")
				headder_list = headder_list_old[:-1]
				headder_list.append(headder_list_old[-1].strip())
				experiment_count = len(headder_list) - 3
				experiment_names = headder_list[-experiment_count:]
				headder = False
			else:
				line_list = line.split("||")
				gene_name_list = line_list[0].split()
				gene_name = gene_name_list[0]

				# Dealing with wrong cases
				gene_name = gene_name.lower()
				#gene_name = gene_name.replace('RV','Rv')
				#gene_name = gene_name.replace('C','c')

				val_list_old = line_list[3].split("\t")
				val_list = val_list_old[1:-1]
				val_list.append(val_list_old[-1].strip())
				#print gene_name, val_list, len(val_list)
				unformatted_data[gene_name] = val_list

		#print unformatted_data

		count = 1 # Starts at one because of the 1 used for scaling as the GWEIGHT
		for experiment in experiment_names:
			experiment_dict = {}
			#print experiment
			for gene in unformatted_data.keys():
				#print gene, unformatted_data[gene][count]
				experiment_dict[gene] = unformatted_data[gene][count]
			count += 1
			training_data.append(experiment_dict)




	# Checks
	#print headder_list
	#print len(headder_list)
	#print experiment_count
	#print experiment_names
	#print training_data[17]['RV0193C']

	return training_data


def mean(a):
    return sum(a) / len(a)

def check_missing_experiments(list_A):
	"For removes the experiments where not all genes were used"
	complete = True
	for element in list_A:
		if len(element) < 1:
			complete = False
	return complete

def get_nn_details(ann):
	for mod in ann.modules:
		for conn in ann.connections[mod]:
			#print conn
			pred_weights = []
			for cc in range(len(conn.params)):
				pred_weights.append(conn.params[cc])
	return pred_weights

def get_sub_list_from_network(origional_network, a_gene, a_label, edge_threshold, *extra_genes):
	"Return a list of genes to be used in training"
	#print full_network.nodes(data=True)
	
	print "Getting input network structure"

	print 'For gene: ' + a_gene
	print 'For connected nodes with label: ' + a_label

	full_network = copy.deepcopy(origional_network)

	#print full_network.node[a_gene]

	for node in full_network.nodes(data=True):
		if node[1]['type'] != a_label and node[0] != a_gene:
			#print node[0]
			full_network.remove_node(node[0])

	#print full_network.nodes(data=True)

	'''
	# Removed for now so not to remove edges below the threshold. all TFs are thus included
	for edge in full_network.edges(data=True):
		
		#print "----->" ,edge
		if edge[2]['fold_change'] < abs(edge_threshold):
			full_network.remove_edge(edge[0],edge[1])
	'''

	#print "------------------"
	sub_graph = 0

	
	all_nodes_list = []

	for con_gene in nx.all_neighbors(full_network, a_gene):
		all_nodes_list.append(str(con_gene))

	#print sub_graph.nodes(data=True)
	#print all_nodes_list
	#print set(all_nodes_list)
	all_nodes_list = list(set(all_nodes_list))

	if len(extra_genes) > 0:
		all_nodes_list.append(extra_genes)

	return all_nodes_list	

	'''

	if full_network[a_gene].degree() > 0:
		sub_graph = nx.ego_graph(full_network, a_gene, undirected=True)
	#print sub_graph.nodes(data=True)
	return sub_graph.nodes()
	'''



def filter_expression_dataset(gene_list, dataset_list, *low_sig_thres):
	""" Primary filtering removes data not needed so the next step is not thrown by other genes values """
	filtered_dataset_list = []
	
	#This fitering step needs work, potentially the 'del' function removes the data for all instances
	# Also needed if subsiquent step of filtering is to have any real effect

	print gene_list

	for experiment in dataset_list:
		list_genes_only_experiment = {}
		for gene in experiment.keys():
			#print gene
			if gene.lower() in gene_list:
				#print gene
				#print experiment[gene]
				list_genes_only_experiment[gene] = experiment[gene]
		filtered_dataset_list.append(list_genes_only_experiment)

	

	#filtered_dataset_list = dataset_list

	#print "+++++++++++++++++++++++"
	#print filtered_dataset_list

	""" Secondary filtering needs revision. perhaps work out the average """
	# Are smaller values better? In pcl file it seems that value 1 is no change, and values to either side are change. Therefore the threshold should be a max rather
	# The above made a massive difference increasing the accuracy of the nets

	if len(low_sig_thres[0]) > 0:
		extra_filtered_dataset_list = []
		#print 'filtering'
		number_of_experiments = 0
		for experiment in filtered_dataset_list:
			total = 0
			num_of_included_readings = 0
			for gene in experiment.keys():
				if len(experiment[gene]) > 0:
					#print "Here"
					#print experiment[gene]
					#print "End"
					abs_experiment = abs(float(experiment[gene]))
					total += abs_experiment
					num_of_included_readings += 1
			print 'total experiment weight'
			print total
			if num_of_included_readings > 0:
				print 'average experiment weight'
				average_exp_weight = total / num_of_included_readings
				print average_exp_weight
				if average_exp_weight < float(low_sig_thres[0]):
					extra_filtered_dataset_list.append(experiment)
					number_of_experiments += 1
		print "number of experiments retained:"
		# test this part out (vary  cutoff, see effect)
		print number_of_experiments
		filtered_dataset_list = extra_filtered_dataset_list


	return filtered_dataset_list



def ANN_blind_analysis_multi_hidden(a_network, a_gene, a_dataset, boot_val, train_for):

	"Creates and trains a network that is created to reflect the structure of the hypothesized network"

	regulatory_network = FeedForwardNetwork()

	# retrieving needed parameters from the input network

	upper_case_data_node_list = get_sub_list_from_network(a_network, a_gene, "TF", 1)

	# to lower case for everything
	data_node_list = [x.lower() for x in upper_case_data_node_list]
	a_gene = a_gene.lower()

	# If the target gene is also a TF, remove it from the list as it will be added
	if a_gene in data_node_list: data_node_list.remove(a_gene)

	print 'What is in data_node_list:'
	print data_node_list

	if len(data_node_list) == 0:
		print "No connections to " + a_gene + " found."
		return [a_gene, '0', '0']


	# Check for missing entries in the dataset (DS)
	# For the main gene

	#print a_gene
	#print a_dataset[0].keys()
	
	# Check for missing entries in the dataset (DS)
    # For the main gene
	if a_gene not in a_dataset[0].keys():
		#print 'herp'
		return [a_gene, '0', '0']
	
	# For the linked genes

	for each_gene in data_node_list:
		if each_gene not in a_dataset[0].keys():
			data_node_list.remove(each_gene)

	if len(data_node_list) == 0:
		print "No connections to " + a_gene + " found."
		return [a_gene, '0', '0']

	print len(data_node_list)
	print data_node_list


	# Need to add +1 node to the input layer that represents the "other" control variables

	# describing network modules to be used
	inLayer = LinearLayer(len(data_node_list), name="Input_layer")
	
	hiddenLayer = SigmoidLayer(len(data_node_list) + 1, name="Hidden_sigmoid_layer_1")

	hiddenLayer2 = SigmoidLayer(len(data_node_list) + 1, name="Hidden_sigmoid_layer_2")
	
	outLayer = LinearLayer(1, name="Output_layer")


	# Adding layers to network
	regulatory_network.addInputModule(inLayer)

	regulatory_network.addModule(hiddenLayer)

	regulatory_network.addModule(hiddenLayer2)

	regulatory_network.addOutputModule(outLayer)

	# Adding connections between layers

	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	
	hidden_to_hidden2 = FullConnection(hiddenLayer, hiddenLayer2)

	hidden2_to_out = FullConnection(hiddenLayer2, outLayer)


	regulatory_network.addConnection(in_to_hidden)

	regulatory_network.addConnection(hidden_to_hidden2)

	regulatory_network.addConnection(hidden2_to_out)


	get_nn_details(regulatory_network)

	# Other stuff added

	regulatory_network.sortModules()

	# Formatting the dataset 

	input_dimention = len(data_node_list)
	print "in_dimention = ", input_dimention

	DS = SupervisedDataSet( input_dimention, 1 )

	# Adding data, there may be a problem with order here where tfs are not always the same... seems ok though


	# This may not be the best way, but is needed due to the next for statement
	data_node_list.append(a_gene)
	print 'node list contains: '
	print data_node_list

	# An additional filtering step to slim down the dataset and remove low signal data -------------------------------------------< FILTER
	a_filtered_dataset = a_dataset

	a_filtered_dataset = filter_expression_dataset(data_node_list, a_filtered_dataset, '0.8')

	# This is where the ordered dict needs to be used to link the input name to the input node.

	for experiment in a_filtered_dataset:
		tf_list = []
		gene_list = []
		tf_labels = []
		first_round = True
		for TF in data_node_list:
			if TF != a_gene:
				#print TF, "<---"
				tf_list.append(experiment[TF])
				if first_round == True:
					tf_labels.append(TF)
			else:
				#print TF, "<---gene"
				gene_list.append(experiment[TF])
		first_round = False
		# View the input data sets
		print tf_labels
		print tf_list
		print gene_list


		if (check_missing_experiments(tf_list) == True) and (check_missing_experiments(gene_list) == True):
			float_tf_list = [float(i) for i in tf_list]
			float_gene_list = [float(i) for i in gene_list]
			DS.appendLinked( float_tf_list, float_gene_list )

	print "......"
	print 'Network before training'
	print regulatory_network

	#pesos_conexiones(regulatory_network)
	print regulatory_network.outputerror

	#print DS

	# Training
	trainer = RPropMinusTrainer_Evolved(regulatory_network, verbose=False)

	trainer.setData(DS)

	result_list = []
	best_run_error = 1000

	train_for = int(train_for)

	boot_count = 0
	while boot_count < boot_val:
		print '\n'
		print 'Bootstrap round ' + str(boot_count + 1)
		trainer.trainEpochs(train_for)
		this = get_nn_details(regulatory_network)
		# Corrected error
		
		print trainer.total_error
		current_run_error = trainer.total_error
		

		
		print 'Bootstrap round ' + str(boot_count + 1) + ' error: ' + str(current_run_error)

		if abs(current_run_error) < abs(best_run_error):
			best_run_error = current_run_error
			trained_net_filename = a_gene + '_trained_net.xml'
			NetworkWriter.writeToFile(regulatory_network, trained_net_filename)

			export_to_gml(regulatory_network, tf_labels, a_gene)

		#result_list.append(this)
		regulatory_network.reset()
		regulatory_network.randomize()
		trainer = RPropMinusTrainer_Evolved(regulatory_network, verbose=False)
		trainer.setData(DS)
		boot_count += 1

	#print "TF Labels"
	#print tf_labels
	#print regulatory_network.params
	#print inLayer
	#print "Pesos Conexiones"
	#pesos_conexiones(regulatory_network)

	#print dir(regulatory_network)
	#print dir(trainer)
	#print 'look here'
	#print regulatory_network.outputerror
	#print '<><><><><>'
	#print dir(regulatory_network['SigmoidLayer-7'])
	#print '\n'
	#print vars(regulatory_network['SigmoidLayer-7'])
	#print '\n'
	#print regulatory_network['SigmoidLayer-7'].forward
	#print regulatory_network['SigmoidLayer-7'].bufferlist

	result_list.append(a_gene)

	result_list.append(best_run_error)

	result_list.append(len(tf_list))

	return result_list

def ANN_blind_analysis(a_network, a_gene, a_dataset, boot_val):

	"Creates and trains a network that is created to reflect the structure of the hypothesized network"

	regulatory_network = FeedForwardNetwork()

	# retrieving needed parameters from the input network

	upper_case_data_node_list = get_sub_list_from_network(a_network, a_gene, "gene,TF", 1)

	# to lower case for everything
	data_node_list = [x.lower() for x in upper_case_data_node_list]
	a_gene = a_gene.lower()

	# If the target gene is also a TF, remove it from the list as it will be added
	if a_gene in data_node_list: data_node_list.remove(a_gene)

	print 'what is in data_node_list:'
	print data_node_list

	if len(data_node_list) == 0:
		print "No connections to " + a_gene + " found."
		return [a_gene, '0', '0']


	# Check for missing entries in the dataset (DS)
	# For the main gene

	#print a_gene
	#print a_dataset[0].keys()
	
	# Check for missing entries in the dataset (DS)
    # For the main gene
	if a_gene not in a_dataset[0].keys():
		#print 'herp'
		return [a_gene, '0', '0']
	
	# For the linked genes

	for each_gene in data_node_list:
		if each_gene not in a_dataset[0].keys():
			data_node_list.remove(each_gene)

	if len(data_node_list) == 0:
		print "No connections to " + a_gene + " found."
		return [a_gene, '0', '0']

	print len(data_node_list)
	print data_node_list

	# Need to add +1 node to the input layer that represents the "other" control variables

	# describing network modules to be used
	inLayer = LinearLayer(len(data_node_list), name="Input_layer")
	
	hiddenLayer = SigmoidLayer(len(data_node_list) + 1, name="Hidden_sigmoid_layer_1")
	
	outLayer = LinearLayer(1, name="Output_layer")


	# Adding layers to network
	regulatory_network.addInputModule(inLayer)

	regulatory_network.addModule(hiddenLayer)

	regulatory_network.addOutputModule(outLayer)

	# Adding connections between layers

	in_to_hidden = FullConnection(inLayer, hiddenLayer)

	hidden_to_out = FullConnection(hiddenLayer, outLayer)


	regulatory_network.addConnection(in_to_hidden)

	regulatory_network.addConnection(hidden_to_out)


	get_nn_details(regulatory_network)

	# Other stuff added

	regulatory_network.sortModules()

	# Formatting the dataset 

	input_dimention = len(data_node_list)
	print "in_dimention = ", input_dimention

	DS = SupervisedDataSet( input_dimention, 1 )

	# Adding data, there may be a problem with order here where tfs are not always the same... seems ok though


	# This may not be the best way, but is needed due to the next for statement
	data_node_list.append(a_gene)
	print 'node list contains: '
	print data_node_list

	# This is where the ordered dict needs to be used to link the input name to the input node.

	for experiment in a_dataset:
		tf_list = []
		gene_list = []
		tf_labels = []
		first_round = True
		for TF in data_node_list:
			if TF != a_gene:
				#print TF, "<---"
				tf_list.append(experiment[TF])
				if first_round == True:
					tf_labels.append(TF)
			else:
				#print TF, "<---gene"
				gene_list.append(experiment[TF])
		first_round = False
		# View the input data sets
		print tf_labels
		print tf_list
		print gene_list


		if (check_missing_experiments(tf_list) == True) and (check_missing_experiments(gene_list) == True):
			float_tf_list = [float(i) for i in tf_list]
			float_gene_list = [float(i) for i in gene_list]
			DS.appendLinked( float_tf_list, float_gene_list )

	print "......"
	print 'Network before training'
	print regulatory_network

	pesos_conexiones(regulatory_network)
	print regulatory_network.outputerror

	#print DS

	# Training
	trainer = RPropMinusTrainer_Evolved(regulatory_network, verbose=False)

	trainer.setData(DS)

	result_list = []
	best_run_error = 1000

	boot_count = 0
	while boot_count < boot_val:
		print '\n'
		print 'Bootstrap round ' + str(boot_count + 1)
		trainer.trainEpochs(500)
		this = get_nn_details(regulatory_network)
		# Corrected error
		
		print trainer.total_error
		current_run_error = trainer.total_error
		

		
		print 'Bootstrap round ' + str(boot_count + 1) + ' error: ' + str(current_run_error)

		if abs(current_run_error) < abs(best_run_error):
			best_run_error = current_run_error
			trained_net_filename = a_gene + '_trained_net.xml'
			NetworkWriter.writeToFile(regulatory_network, trained_net_filename)

			export_to_gml(regulatory_network, tf_labels, a_gene)

		#result_list.append(this)
		regulatory_network.reset()
		regulatory_network.randomize()
		trainer = RPropMinusTrainer_Evolved(regulatory_network, verbose=False)
		trainer.setData(DS)
		boot_count += 1

	#print "TF Labels"
	#print tf_labels
	#print regulatory_network.params
	#print inLayer
	#print "Pesos Conexiones"
	#pesos_conexiones(regulatory_network)

	#print dir(regulatory_network)
	#print dir(trainer)
	#print 'look here'
	#print regulatory_network.outputerror
	#print '<><><><><>'
	#print dir(regulatory_network['SigmoidLayer-7'])
	#print '\n'
	#print vars(regulatory_network['SigmoidLayer-7'])
	#print '\n'
	#print regulatory_network['SigmoidLayer-7'].forward
	#print regulatory_network['SigmoidLayer-7'].bufferlist

	result_list.append(a_gene)

	result_list.append(best_run_error)

	result_list.append(len(tf_list))

	return result_list


def ANN_edge_analysis(a_network, a_gene, a_dataset, boot_val):

	"Creates and trains a network that is created to reflect the structure of the hypothesized network"

	regulatory_network = FeedForwardNetwork()

	# retrievingneeded parameters from the input network

	data_node_list = get_sub_list_from_network(a_network, a_gene, "gene,TF", 1)

	# Need to add +1 node to the input layer that represents the "other" control variables

	# describing network modules to be used
	inLayer = LinearLayer(len(data_node_list)-1)
	#hiddenLayer = LinearLayer(len(data_node_list)-1))
	outLayer = LinearLayer(1)


	# Adding layers to network
	regulatory_network.addInputModule(inLayer)
	#regulatory_network.addModule(hiddenLayer)
	regulatory_network.addOutputModule(outLayer)

	# Adding connections between layers

	#in_to_hidden = LinearConnection(inLayer,hiddenLayer)
	#hidden_to_out = FullConnection(hiddenLayer, outLayer)

	in_to_out = FullConnection(inLayer, outLayer)

	#regulatory_network.addConnection(in_to_hidden)
	#regulatory_network.addConnection(hidden_to_out)

	regulatory_network.addConnection(in_to_out)

	get_nn_details(regulatory_network)

	# Other stuff added

	regulatory_network.sortModules()

	# Formatting the dataset 

	input_dimention = len(data_node_list)-1
	print "in_dimention = ", input_dimention

	DS = SupervisedDataSet( input_dimention, 1 )

	# Adding data, there may be a problem with order here where tfs are not always the same... seems ok though

	for experiment in a_dataset:
		tf_list = []
		gene_list = []
		tf_labels = []
		for TF in data_node_list:
			if TF != a_gene:
				#print TF, "<---"
				tf_list.append(experiment[TF])
				tf_labels.append(TF)
			else:
				#print TF, "<---gene"
				gene_list.append(experiment[TF])

		print tf_list
		print gene_list


		if (check_missing_experiments(tf_list) == True) and (check_missing_experiments(gene_list) == True):
			float_tf_list = [float(i) for i in tf_list]
			float_gene_list = [float(i) for i in gene_list]
			DS.appendLinked( float_tf_list, float_gene_list )

	print "......"

	print DS

	# Training
	trainer = BackpropTrainer(regulatory_network, momentum=0.1, verbose=True, weightdecay=0.01)

	trainer.setData(DS)

	result_list = []

	boot_count = 0
	while boot_count < boot_val:
		#trainer.trainEpochs(1000)
		trainer.trainUntilConvergence(validationProportion=0.25)
		print regulatory_network
		this = get_nn_details(regulatory_network)
		result_list.append(this)
		regulatory_network.reset()
		boot_count += 1

	print tf_labels
	print regulatory_network.params
	print in_to_out.params
	print inLayer
	pesos_conexiones(regulatory_network)

	NetworkWriter.writeToFile(regulatory_network, 'trained_net.xml')
	return result_list

def export_to_gml(network_obj, gene_list_in_order, out_node):
	'''Does what it says on the box'''
	# Get info from network object (Grab from things like "pesos_conexiones(regulatory_network)")
	# Get error somehow...
	# Make a networkx netowrk then export? Lets try that.
	# ADD the weights / values of the hidden node transfer function!!!!

	#print 'Exporiting to gml'

	ann_nx_graph = nx.DiGraph()
	
	# Looking at the network
	#print 'Net details'
	#print type(network_obj)
	#print dir(network_obj)

	for mod in network_obj.modules:
		#print dir(mod)
		#if mod.paramdim > 0:
		#	print 'HAVE PARAMS'
		#	print mod.params
		#print mod.name
		#print gene_list_in_order

		for conn in network_obj.connections[mod]:
			

			# Extract layer names
			layer_list = str(conn).split(' ')
			from_layer = layer_list[2][1:-1]
			to_layer = layer_list[4][1:-2]
			print from_layer , ' ', to_layer


			for cc in range(len(conn.params)):
			
				if mod.name == 'Input_layer':
					#print 'Input_layer'
					#print conn.whichBuffers(cc), conn.params[cc]
					#print conn.whichBuffers(cc)[0], conn.whichBuffers(cc)[1]
					from_node_name = gene_list_in_order[conn.whichBuffers(cc)[0]]
					to_node_name = to_layer + '_' + str(conn.whichBuffers(cc)[1])
					#print conn.params


				elif to_layer == 'Output_layer':
					from_node_name = from_layer + '_' + str(conn.whichBuffers(cc)[0])
					to_node_name = out_node
					#print conn.params
					#print 'Output_layer'
					#print conn.whichBuffers(cc), conn.params[cc]
					#print conn.whichBuffers(cc)[0], conn.whichBuffers(cc)[1]
					# Connections not needed, it's the last layer
					#from_node_name = from_layer + '_' + str(conn.whichBuffers(cc)[0])
					#to_node_name = out_node

				else:
					#print 'Hidden_layer'
					#print conn.whichBuffers(cc), conn.params[cc]
					#print conn.whichBuffers(cc)[0], conn.whichBuffers(cc)[1]
					#to_node_name = out_node
					from_node_name = from_layer + '_' + str(conn.whichBuffers(cc)[0])
					to_node_name = to_layer + '_' + str(conn.whichBuffers(cc)[1])
					#print conn.params

				# Add edges
				ann_nx_graph.add_weighted_edges_from([(from_node_name,to_node_name,conn.params[cc])])
		is_input_layer = False

	gml_file_name = out_node + '_trained_net_thres.gml'

	print 'Exporting to gml'
	nx.write_gml(ann_nx_graph, gml_file_name)


def plot_gene_vs_TF(dataset, q_gene, q_tf):
	'''Lets just have a look'''
	
	x = []
	y = []

	for experiment in dataset:
		if len(experiment[q_gene]) > 1 and len(experiment[q_tf]) > 1:
			x.append(float(experiment[q_gene]))
			y.append(float(experiment[q_tf]))
	
	plt.scatter(x, y)

	plt.show()

def create_optimized_network(organism_network, shared_dataset, bootstrap_value, train_for, result_file, *gene_list_path):
	'''Analyze all genes in a network and create ANNs. Outouts are a gml and xml per gene, and a report csv file with per gene info on errors etc'''
	
	# Output csv of the errors and the number of input nodes


	summary_file = open(result_file, 'w')

	# First, get a list of the genes to be analysed. Deciding which should be included based on a label such as 'type'
	list_of_nodes_for_analysis = []

	# What the label must contain
	label_val = 'gene'

	if len(gene_list_path[0]) > 0:
		gene_list_file = open(gene_list_path[0], 'r')
		
		# Filter full list based on label criteria
		for line in gene_list_file:
			line = line.strip('\n')
			if label_val in organism_network.node[line]['type']:
				list_of_nodes_for_analysis.append(line)

	else:
		# What the label must contain
		full_list = organism_network.nodes()

		# Filter full list based on label criteria
		for gene_node in full_list:
			if label_val in organism_network.node[gene_node]['type']:
				list_of_nodes_for_analysis.append(gene_node)

	print list_of_nodes_for_analysis

	initial_shared_dataset = shared_dataset

	for target_gene_node in list_of_nodes_for_analysis:
		instance_dataset = initial_shared_dataset
		print "Current gene: ", target_gene_node

		analysis_output = ANN_blind_analysis_multi_hidden(organism_network, target_gene_node, instance_dataset, bootstrap_value, train_for)
		print analysis_output
		writen_line = target_gene_node + ',' + str(analysis_output[1]) + ',' + str(analysis_output[2]) + '\n'
		print writen_line
		summary_file.write(writen_line)

	summary_file.close()

	print "done"

def create_optimized_network_parra(organism_network, shared_dataset, bootstrap_value, result_file):
	'''Analyze all genes in a network and create ANNs. Outouts are a gml and xml per gene, and a report csv file with per gene info on errors etc'''
	

	from Queue import Queue
	import threading

	# Output csv of the errors and the number of input nodes

	summary_file = open(result_file, 'w')

	# First, get a list of the genes to be analysed. Deciding which should be included based on a label such as 'type'
	list_of_nodes_for_analysis = []

	# What the label must contain
	label_val = 'gene'

	full_list = organism_network.nodes()

	# Filter full list based on label criteria

	for gene_node in full_list:
		if label_val in organism_network.node[gene_node]['type']:
			list_of_nodes_for_analysis.append(gene_node)


	# Multithreading experiments
	
	'''
	q = Queue(maxsize=2)

	# Actual analysis part ()


	for target_gene_node in list_of_nodes_for_analysis:
		q.put(ANN_blind_analysis(organism_network, target_gene_node, shared_dataset, bootstrap_value))




	thread_result = []
	count = 0

	while count < len(list_of_nodes_for_analysis):
		thread_result.append(q.get())
		count += 1

	q.task_done() 
	print 'Thread output is:'
	print thread_result

	#summary_file.close()
	'''
	import multiprocessing as mp

	results = []

	pool = mp.Pool(processes=4)


	for target_gene_node in list_of_nodes_for_analysis:
			results = pool.apply(ANN_blind_analysis, args=(organism_network, target_gene_node, shared_dataset, bootstrap_value))


	print "This is the output"
	print(results)

	print "done"



# -------------------------  Known Issues  -------------------------
'''
1. Different formatting for gene names ("Rv" vs "RV") (Temp fix line 114)
	Also applies to the "c" at the end... hmmmm
2. Add reseting of the network (Done)
3. Add activate function (Done in class implimentation)
4. Make sure the network reset is not blanking the network before printing the results (Seems ok)
5. Are we using the right input value in the .pcl file? Maybe the zscore is not the best...
6. Remove data where all nodes have 0.0 score and no training value
7. Current version does not take direction of connection into account it seems
'''

# -----------------------  Future features  -------------------------
'''
1. create a python script / function that using a gml, computes output based on input. To facilitate inclusion in pipelines downstream.
2. Add additional hidden layers untill the error stops decreasing. (Adaptive neural network)
3. look at http://pybrain.org/docs/api/supervised/trainers.html --> RPropMinusTrainer (Implemented)
4. Use python ordered dict !!!
'''

# -------------------------  Working Area ------------------------- 



print "importing dataset"

#input_dataset = import_training_data("../experimental_data/Rv1934c_1_25.pcl")

input_dataset = import_training_data('/Volumes/HDD/Genomes/M_tuberculosis/H37Rv/expression_data/Rv1934c_1_25.pcl')

print "importing dataset - complete"

print "Loading network"

#H37Rv_TF_network = nx.read_gml('../networks/S507_S5537_noPPI_net.gml', relabel=True)
H37Rv_TF_network = nx.read_gml('/Users/panix/Dropbox/Programs/tools/Cell/Cell_core/H37Rv_TF_only.gml', relabel=True)

print "Loading network - complete"

print "Starting analysis"

N_H_Layers = 2

create_optimized_network(H37Rv_TF_network, input_dataset, 6, 100, 'filter_test_run.csv', 'xaa')

print "Analysis complete"

print "tock"



# -------------------------  Testing Area ------------------------- 
'''


# For testing the recovery of info from generated ANNs

Rv1990c_path = "/Volumes/HDD/Genomes/M_tuberculosis/H37Rv/h37rv_ANN/neural_cell/fullrun_3/rv1990c_trained_net.xml"
Rv1990c_gml_path = "/Volumes/HDD/Genomes/M_tuberculosis/H37Rv/h37rv_ANN/neural_cell/fullrun_3/rv1990c_trained_net_thres.gml"
#a_dataset = import_training_data("/Volumes/HDD/Genomes/M_tuberculosis/H37Rv/expression_data/Rv2429.pcl")
#a_dataset = import_training_data("test.pcl")
#test_g = nx.read_gml('test.gml', relabel=True)



#print H37Rv_TF_network.node[1]

#something = get_sub_list_from_network(test_g, "RV1026", "TF", 1)

#print ANN_edge_analysis(test_g, "RV1026", a_dataset)

#map(mean, zip(*result_list))

print ""

# Expression data

print "importing dataset"
input_dataset = import_training_data("/Volumes/HDD/Genomes/M_tuberculosis/H37Rv/expression_data/Rv1934c_1_75.pcl")


print "importing dataset - complete"

# Plots showing TF - gene relationships

#plot_gene_vs_TF(a_dataset, "Rv3133c", "Rv2626c")
#plot_gene_vs_TF(a_dataset, "Rv3133c", "Rv2005c")
#plot_gene_vs_TF(a_dataset, "Rv3249c", "Rv1917c")
#plot_gene_vs_TF(a_dataset, "Rv0081", "Rv2491")

#plot_gene_vs_TF(a_dataset, "Rv3128c", "Rv3133c")
#plot_gene_vs_TF(a_dataset, "Rv3128c", "Rv2034")
#plot_gene_vs_TF(a_dataset, "Rv3128c", "Rv3574")



# Network
print "Loading network"
#H37Rv_TF_network = nx.read_gml('rv1587c_trained_net_thres.gml', relabel=True)

#H37Rv_TF_network = nx.read_gml('/Users/panix/Dropbox/Programs/tools/Cell/Cell_core/S507_S5537_noPPI_net.gml', relabel=True)

#H37Rv_TF_network = nx.read_gml('/Users/panix/Dropbox/Programs/tools/Cell/Cell_core/TF_network.gml',  relabel=True)

RV1990c_net = gene_Neuron_Cluster('rv1990c_trained_net_thres.gml')
RV1990c_net.add_xml_file('rv1990c_trained_net.xml')
print RV1990c_net.predict_output([1.59,0.60,-0.28])

print RV1990c_net.input_list()




print "Loading network - complete"


RV1026 = ANN_blind_analysis_multi_hidden(test_g, "RV1026", bias_training_data, 4, 1000)


exp_dataset_5["rv0102"] = '2'
exp_dataset_5["rv3056"] = '6'
exp_dataset_5["rv0912"] = '10'
exp_dataset_5["rv0007"] = '4'
exp_dataset_5["rv1026"] = '46'


exp_dataset_4["rv0102"] = '2'
exp_dataset_4["rv3056"] = '6'
exp_dataset_4["rv0912"] = '4'
exp_dataset_4["rv0007"] = '4'
exp_dataset_4["rv1026"] = '19'



RV1026_net = gene_Neuron_Cluster('rv1026_trained_net_thres.gml')
RV1026_net.add_xml_file('rv1026_trained_net.xml')

print RV1026_net.predict_output([4,6,10,2])
# Expect 46

print RV1026_net.predict_output([4,6,4,2])
# Expect 19

print RV1026_net.input_list()

#print 'returned result'
#print RV1026
# Analysis



print "Starting analysis"

# Testing relabeling of data inputs
#RV1587c = ANN_blind_analysis(H37Rv_TF_network, "Rv1587c", input_dataset, 2)

# Test where gene has one TF
#Rv2626c = ANN_blind_analysis(H37Rv_TF_network, "Rv2626c", input_dataset, 2)
# Test where the gene is a TF
#Rv3133c = ANN_blind_analysis(H37Rv_TF_network, "Rv3133c", input_dataset, 2)
# Test where gene has 2 TF
#Rv2763c = ANN_blind_analysis(H37Rv_TF_network, "Rv2763c", input_dataset, 2)

# No more error
#Rv1934c = ANN_blind_analysis(H37Rv_TF_network, "Rv1934c", input_dataset, 2)

# latest error
#Rv0178 = ANN_blind_analysis(H37Rv_TF_network, "Rv0178", input_dataset, 2)

#print Rv0178
N_H_Layers = 2

#create_optimized_network_parra(test_g, bias_training_data, 2, 'test_run_T.csv')

#create_optimized_network(H37Rv_TF_network, input_dataset, 6, 'test_run_2.csv')

#create_optimized_network_parra(H37Rv_TF_network, input_dataset, 2, 'test_run_0.csv')

print "Analysis complete"

#print Rv2626c
#print Rv3133c
#print Rv1934c

print "tock"
'''
