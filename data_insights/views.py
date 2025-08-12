from django.shortcuts import render

def data_insights(request):
    return render(request, 'data_insights/datainsights.html')


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "")

        # Example chatbot logic — replace this with real AI model call
        bot_response = f"You said: {user_message}"

        return JsonResponse({"response": bot_response})

    return JsonResponse({"error": "Invalid request"}, status=400)
