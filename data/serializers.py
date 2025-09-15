from rest_framework.serializers import ModelSerializer
from .models import NoiseDataset,Dataset
from core.serializers import Microphone_TypeSerializer,Time_Of_DaySerializer,SubClassSerializer,CategorySerializer,ClassSerializer,CommunitySerializer,RegionSerializer


class DatasetSerializer(ModelSerializer):
    class Meta:
        model = Dataset
        fields = ["id","name"]

        
class NoiseDatasetSerializer(ModelSerializer):
    region = RegionSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    time_of_day = Time_Of_DaySerializer(read_only=True)
    community = CommunitySerializer(read_only=True)
    class_name = ClassSerializer(read_only=True)
    subclass = SubClassSerializer(read_only=True)
    microphone_type = Microphone_TypeSerializer(read_only=True)
    dataset_type = DatasetSerializer(read_only=True)
    class Meta:
        model = NoiseDataset
        fields = '__all__'




