from rest_framework.serializers import ModelSerializer
from .models import Class,SubClass,Category,Microphone_Type,Time_Of_Day,Community,Region



class ClassSerializer(ModelSerializer):
    class Meta:
        model = Class
        fields = ['id','name']


class SubClassSerializer(ModelSerializer):
    class Meta:
        model = SubClass
        fields = ['id','name']


class CategorySerializer(ModelSerializer):
    class Meta:
        model = Category
        fields = ['id','name']


class Microphone_TypeSerializer(ModelSerializer):
    class Meta:
        model = Microphone_Type
        fields = ['id','name']



class Time_Of_DaySerializer(ModelSerializer):
    class Meta:
        model = Time_Of_Day
        fields = ['id','name']



class CommunitySerializer(ModelSerializer):
    class Meta:
        model = Community
        fields = ['id','name']


class RegionSerializer(ModelSerializer):
    class Meta:
        model = Region
        fields = ['id','name']

