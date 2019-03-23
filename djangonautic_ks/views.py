from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.contrib.auth import login,logout
from mainpage.models import user_list,ParametersDB,ParametersDB1,KeystrokeDB1,voiceFeatures
import sqlite3
import pandas as pd  
import numpy as np  
from . import my_settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import json
import re

name=''
iid=-1
sentence=['qwerty']
char_list=[]
index=0
ch_index=0

def homepage(request):
	return render(request,'homepage.html')

def signup_view(request):
	global index

	if request.method == 'POST':
		
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			#log the user in
			login(request, user)
			print("in Signup post")
			name=user_list.objects.create(user_name=str(user), index=0)
			name.save()

			
			return redirect('mainpage:mainpage')
	else:
		form = UserCreationForm()
	return render(request, 'Signup.html',{'form':form})

def login_view(request):
	
	return render(request,'Login.html',{ 'data':sentence[0]})

def update1(request):
	
	global index
	global ch_index
	global name
	global iid
	file=''
	table=''
	text=''
	text1='correct user'
	text2='incorrect user'
	if request.method == 'POST':
		if 'dict' in request.POST:
			#print(request.POST)
			#print(time_list)
			time_list= request.POST.get('dict') 

     
			print(time_list,'timelist')
			print(name,'name...........')

			length=int(request.POST.get('length'))
			print(length,'length................')
			time_list= json.loads(time_list)
			instance=KeystrokeDB1.objects.get_or_create(user_name=str(name), user_id=iid, sentence_index=0)
			instance=KeystrokeDB1.objects.get(user_name=str(name),user_id=iid, sentence_index=0)
			key_list = list(time_list.keys())
			for i in range (0, length):
				key_split1=re.split('[0-9]+',key_list[i])
				#print(key_split1)
				if(key_split1[2]=='dn'):
					if(key_split1[1]=='Backspace'):
						char_list.append(8)
					elif(key_split1[1]=='Shift'):
						char_list.append(16)
					elif(key_split1[1]=="CapsLock"):
						char_list.append(20)
					elif(key_split1[1]=="Tab"):
						char_list.append(9)
					elif(key_split1[1]=="Control"):
						char_list.append(17)
					elif(key_split1[1]=="Alt"):
						char_list.append(18)
					elif(key_split1[1]=='ArrowLeft'):
						char_list.append(37)
					elif(key_split1[1]=='ArrowRight'):
						char_list.append(39)
					elif(key_split1[1]=='Enter'):
						char_list.append(13)
					else:
						char_list.append(ord(key_split1[1]))

					#print('in down',key_split1[1])
					#print('in down if',key_split1[1])
					temp=time_list[key_list[i]]
					exec("instance.chr%d_dn=temp" % ch_index)
					for j in range(i,length):
						key_split2=re.split('[0-9]+',key_list[j])
						if(key_split2[2]=='up'):
						#print('in up',key_split2[1])
							if(key_split1[1] == key_split2[1]):
								exec("instance.chr%d_up= time_list[key_list[j]]" % ch_index)
								ch_index=ch_index+1
								break

			instance.save()
			i=0

			print(char_list)
			for i in range (0, ch_index-1):

				parameters=ParametersDB1.objects.create(user_name=str(name),user_id=iid, char_index = i, sentence_index=0)
				parameters=ParametersDB1.objects.get(user_name=str(name),user_id=iid, char_index = i, sentence_index=0)
				parameters.char_1=char_list[i]
				parameters.char_2= char_list[i+1]
				exec("parameters.DD= instance.chr%d_dn-instance.chr%d_dn" % (i+1,i))
				exec("parameters.UD= instance.chr%d_dn-instance.chr%d_up" % (i+1,i))
				exec("parameters.DU= instance.chr%d_up-instance.chr%d_dn" % (i+1,i))
				exec("parameters.UU= instance.chr%d_up-instance.chr%d_up" % (i+1,i))
				exec("parameters.H1= instance.chr%d_up-instance.chr%d_dn" % (i,i))
				exec("parameters.H2= instance.chr%d_up-instance.chr%d_dn" % (i+1,i+1))
				i=i+1
				parameters.save()
				#print('sentence index...........',sentence_index)
			file='test_ks.csv'
			table='mainpage_parametersdb1'
			convert_modeltocsv(iid,iid,file,table)
			
			ParametersDB1.objects.filter(user_id=iid).delete()
			text=Random_forest()
			

			char_list.clear()
			ch_index=0
			name=''
			iid=-1
			print(text,'.........................................text')

			print(text==text1)
		if (text==text1):	
#		if (text==text1 and text_v==text1):
			print("in iffffffffffffffffff")
			response=0
			return HttpResponse(response)
		elif(text==text2):
			response=1
			return HttpResponse(response)
	else:
		return HttpResponse('fail')

def logout_view(request):
	if request.method=='POST':
		logout(request)
		return render(request, 'homepage.html')





def Authentication_view(request):
	
	return render(request,'Authentication.html')

def Authentication_view1(request):
	
	return render(request,'Authentication1.html')
def log_view(request):

	global index
	global name
	global iid
	id1=-1
	id2=-1
	file=''
	table=''
	print(request.POST,'..................................request')
	if request.method == 'POST':
		#print(request,'..................................request')
		if 'username' in request.POST:
			username= request.POST.get('username')
			name= username
			instance=ParametersDB.objects.get(user_name=str(username), sentence_index=0, char_index=0 )
			features=voiceFeatures.objects.get(user_name=str(username), frame_index=0 )
			if((instance or features) is not None):
				iid=instance.user_id
				
				if(iid<5):
					id1=0
					id2=iid
				else:
					id1=iid-5
					id2=iid
				file='train_ks.csv'
				table='mainpage_parametersdb'
				convert_modeltocsv(id1,id2,file,table)
				file='train_v.csv'
				table='mainpage_voicefeatures'
				convert_modeltocsv_voice(id1,id2,file,table)
				return redirect('Login')
			else:
				print("invalid")

		    

			
	else:
		form = AuthenticationForm()
	return render(request,'log.html')




def convert_modeltocsv(id1,id2,file,table):
	with open(file,'w+') as write_file:
		conn = sqlite3.connect('db.sqlite3')
		att_row='id'+','+'user_id'+','+'user_name'+','+'sentence_index'+','+'char_index'+','+'char_1'+','+'char_2'+','+'DD'+','+'DU'+','+'UD'+','+'UU'+','+'H1'+','+'H2'+'\n'
		cursor=conn.cursor()

		write_file.write(att_row)

		
		for row in cursor.execute('SELECT * FROM {} WHERE (user_id <= ?) AND (user_id >= ?) '.format(table),(id2,id1)):
			temp_row=str(row[0])+','+str(row[1])+','+str(row[2])+','+str(row[3])+','+str(row[4])+','+str(row[5])+','+str(row[6])+','+str(row[7])+','+str(row[8])+','+str(row[9])+','+str(row[10])+','+str(row[11])+','+str(row[12])+'\n'

			write_file.write(temp_row)

def Random_forest():
	
	dataset = pd.read_csv("train_ks.csv")
	dataset_tst =pd.read_csv("test_ks.csv")
	print(dataset)
	X_train = dataset.iloc[:, 5:13].values  
	y_train = dataset.iloc[:,2].values  
	X_test = dataset_tst.iloc[:, 5:13].values 
	y_test = dataset_tst.iloc[:, 2].values 
	regressor = RandomForestClassifier(n_estimators=203,criterion='gini', random_state=0,bootstrap=True, class_weight=None,
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_jobs=None,
            oob_score=True, verbose=0, warm_start=False)  
	regressor.fit(X_train, y_train)  
	y_pred = regressor.predict(X_test) 
	conf_matrix=confusion_matrix(y_test,y_pred)  
#	print(conf_matrix)

#	print(accuracy_score(y_test, y_pred))  
	precision,recall,fscore,support= precision_recall_fscore_support(y_test,y_pred)
	
	conf_matrix=np.array(conf_matrix)
	conf_shape=np.shape(conf_matrix)
	zero_row=np.where(~conf_matrix.any(axis=1))[0]
	for i in range (len(conf_matrix)):
		if i  not in zero_row:
			if(np.argmax(conf_matrix[i])==i):
				return 'correct user'
			else:
				return 'incorrect user'


def convert_modeltocsv_voice(id1,id2,file,table):
	with open(file,'w+') as write_file:
		conn = sqlite3.connect('db.sqlite3')
		att_row='user_id'+','+'user_name'+','+'frame_index'+','+'f0'+','+'f1'+','+'f2'+','+'f3'+','+'f4'+','+'f5'+','+'f6'+','+'f7'+','+'f8'+','+'f9'+','+'f10'+','+'f11'+','+'f12'+','+'f13'+','+'f14'+','+'f15'+','+'f16'+','+'f17'+','+'f18'+','+'f19'+','+'f20'+','+'f21'+','+'f22'+','+'f23'+','+'f24'+','+'f25'+','+'f26'+','+'f27'+','+'f28'+','+'f29'+','+'f30'+','+'f31'+','+'f32'+','+'f33'+','+'f34'+','+'f35'+','+'f36'+','+'f37'+','+'f38'+','+'f39'+','+'f40'+','+'f41'+','+'f42'+','+'f43'+','+'f44'+','+'f45'+','+'f46'+','+'f47'+','+'f48'+','+'f49'+','+'f50'+','+'f51'+','+'f52'+','+'f53'+','+'f54'+','+'f55'+','+'f56'+','+'f57'+','+'f58'+','+'f59'+','+'f60'+','+'f61'+','+'f62'+','+'f63'+','+'f64'+','+'f65'+','+'f66'+','+'f67'+','+'f68'+','+'f69'+','+'f70'+','+'f71'+','+'f72'+','+'f73'+','+'f74'+','+'f75'+','+'f76'+','+'f77'+','+'f78'+','+'f79'+','+'f80'+','+'f81'+','+'f82'+','+'f83'+','+'f84'+','+'f85'+'\n'
		cursor=conn.cursor()

		write_file.write(att_row)

		
		for row in cursor.execute('SELECT * FROM {} WHERE (user_id <= ?) AND (user_id >= ?) '.format(table),(id2,id1)):
			temp_row=str(row[0])+','+str(row[1])+','+str(row[2])+','+str(row[3])+','+str(row[4])+','+str(row[5])+','+str(row[6])+','+str(row[7])+','+str(row[8])+','+str(row[9])+','+str(row[10])+','+str(row[11])+','+str(row[12])+','+str(row[13])+','+str(row[14])+','+str(row[15])+','+str(row[16])+','+str(row[17])+','+str(row[18])+','+str(row[19])+','+str(row[20])+','+str(row[21])+','+str(row[22])+','+str(row[23])+','+str(row[24])+','+str(row[25])+','+str(row[26])+','+str(row[27])+','+str(row[28])+','+str(row[29])+','+str(row[30])+','+str(row[31])+','+str(row[32])+','+str(row[33])+','+str(row[34])+','+str(row[35])+','+str(row[36])+','+str(row[37])+','+str(row[38])+','+str(row[39])+','+str(row[40])+','+str(row[41])+','+str(row[42])+','+str(row[43])+','+str(row[44])+','+str(row[45])+','+str(row[46])+','+str(row[47])+','+str(row[48])+','+str(row[49])+','+str(row[50])+','+str(row[51])+','+str(row[52])+','+str(row[53])+','+str(row[54])+','+str(row[55])+','+str(row[56])+','+str(row[57])+','+str(row[58])+','+str(row[59])+','+str(row[60])+','+str(row[61])+','+str(row[62])+','+str(row[63])+','+str(row[64])+','+str(row[65])+','+str(row[66])+','+str(row[67])+','+str(row[68])+','+str(row[69])+','+str(row[70])+','+str(row[71])+','+str(row[72])+','+str(row[73])+','+str(row[74])+','+str(row[75])+','+str(row[76])+','+str(row[77])+','+str(row[78])+','+str(row[79])+','+str(row[80])+','+str(row[81])+','+str(row[82])+','+str(row[83])+','+str(row[84])+','+str(row[85])+','+str(row[86])+','+str(row[87])+','+str(row[88])+','+str(row[89])+'\n'
			write_file.write(temp_row)

