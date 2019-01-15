from django.shortcuts import render,redirect
from .models import user_list,KeystrokeDB
from django.http import HttpResponse,JsonResponse
import time

# Create your views here.
sentence = ['sentence1','sentence2','sentence3']
def mainpage_view(request):
	username=request.user.username
	name=user_list.objects.get(user_name=str(username))
	index=name.index
	if request.method=='POST':
		#validat

		index=index+1
		name.index=index
		name.save()
		print("...................................in mainpage POST index",index)

		if index<3:
			return redirect('mainpage:mainpage') 
		else:
			print(".....................in else of mainpage POST")

			return redirect('thankyou')
	else:
		print("...................................in mainpage GET index",index)
		if(index<3):
			instance=KeystrokeDB.objects.get_or_create(user_name=str(username), sentence_index=index)

			return render(request, 'mainpage/mainpage.html',{'data':sentence[index], 'instance':instance})
		else:
			return redirect('thankyou')

def update(request):
	i=0
	j=0
	time_list=[]
	if request.method=='POST':
		if 'time_list[]' in request.POST:
			#print(request.POST)
			#print(time_list)
			time_list= request.POST.getlist('time_list[]')
			
			time_list = [int(x) for x in time_list]
			#print(time_list)
			username=request.user.username
			name=user_list.objects.get(user_name=str(username))
			sentence_index=name.index
			instance=KeystrokeDB.objects.get(user_name=str(username), sentence_index=sentence_index)
			list_index=request.POST.getlist('list_index')
			list_index=int(list_index[0])
#			expected_list_index=2*len(sentence[sentence_index])
			
			while(i<list_index):
				exec("instance.chr%d_dn= time_list[i]" % j)

				i=i+1
				exec("instance.chr%d_up= time_list[i]" % j)
				j=j+1
				i=i+1
#			exec("instance.chr%c_up=(time.time()*1000)" %char_index)
			instance.save()
			
			return HttpResponse('success')
	else:
		return HttpResponse('fail')	






