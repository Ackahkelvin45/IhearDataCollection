from rest_framework import serializers
from .models import ChatMessage, ChatSession
from django.utils import timezone
from django.core.validators import MinLengthValidator
import re


class ChatMessageListSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "user_input",
            "assistant_response",
            "created_at",
            "status",
            "visulization",
        ]


class ChatMessageDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "user_input",
            "assistant_response",
            "created_at",
            "status",
            "visulization",
            "tool_called",
            "updated_at",
        ]


class ChatMessageCreateSerializer(serializers.Serializer):
    user_input = serializers.CharField(
        max_length=10000,
        validators=[MinLengthValidator(1)],
        help_text="The user's question or request",
        trim_whitespace=True,
    )
    ai_answer = serializers.BooleanField(
        default=False, help_text="Whether AI should provide interpretation of results"
    )

    def validate_user_input(self, value):
        value = re.sub(r"\s+", " ", value.strip())

        dangerous_patterns = [
            r"(?i)(drop|delete|truncate|alter)\s+table",
            r"(?i)exec\s*\(",
            r"(?i)union\s+select",
            r"--\s*$",
            r"/\*.*?\*/",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, value):
                raise serializers.ValidationError(
                    "Input contains potentially harmful content. Please rephrase your question."
                )

        return value

    def create(self, validated_data):
        raise NotImplementedError("Use ChatMessageCreateSerializer for validation only")


