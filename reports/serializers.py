from rest_framework import serializers



class ReportFilterSerializer(serializers.Serializer):
    categories =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    classes =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    subclasses =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    regions =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    communities =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    microphones =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    timeOfDay =serializers.ListField(
        child=serializers.IntegerField(),required=False
    )
    recordingDevice =serializers.CharField(required=False,allow_blank=True)
    dateRange =serializers.CharField(required=False,allow_blank=True)