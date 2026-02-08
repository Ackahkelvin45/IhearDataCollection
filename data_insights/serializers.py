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
    mode = serializers.ChoiceField(
        choices=["analysis", "ml"],
        required=False,
        help_text="Session mode: analysis or ml",
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


class ChatSessionUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = [
            "title",
            "status",
        ]

    def validate_status(self, value):
        if self.instance:
            current_status = self.instance.status

            allowed_transitions = {
                ChatSession.Status.ACTIVE: [ChatSession.Status.ARCHIVED],
                ChatSession.Status.ARCHIVED: [ChatSession.Status.ACTIVE],
                ChatSession.Status.DELETED: [],
            }

            if value not in allowed_transitions.get(current_status, []):
                raise serializers.ValidationError(
                    f"Cannot transition from {current_status} to {value}"
                )

        return value

    def update(self, instance, validated_data):
        if validated_data.get("status") == ChatSession.Status.ARCHIVED:
            validated_data["archived_at"] = timezone.now()
        elif validated_data.get("status") == ChatSession.Status.ACTIVE:
            validated_data["archived_at"] = None

        return super().update(instance, validated_data)


class MessageStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ["id", "status", "processing_time_ms"]
        read_only_fields = fields


class ChatSessionDetailSerializer(serializers.ModelSerializer):
    messages = ChatMessageListSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "title",
            "mode",
            "status",
            "total_messages",
            "created_at",
            "updated_at",
            "archived_at",
            "messages",
        ]
        read_only_fields = [
            "id",
            "total_messages",
            "created_at",
            "updated_at",
            "archived_at",
            "messages",
        ]


class ChatSessionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = [
            "id",
            "title",
            "mode",
        ]
        read_only_fields = ["id"]

    def validate_title(self, value):
        # Title is now optional - will be generated from first message
        if value:
            value = value.strip()
            if len(value) < 3:
                raise serializers.ValidationError(
                    "Title must be at least 3 characters long"
                )
        return value


class ChatSessionUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = [
            "title",
            "status",
        ]

    def validate_status(self, value):
        if self.instance:
            current_status = self.instance.status

            allowed_transitions = {
                ChatSession.Status.ACTIVE: [ChatSession.Status.ARCHIVED],
                ChatSession.Status.ARCHIVED: [ChatSession.Status.ACTIVE],
                ChatSession.Status.DELETED: [],
            }

            if value not in allowed_transitions.get(current_status, []):
                raise serializers.ValidationError(
                    f"Cannot transition from {current_status} to {value}"
                )

        return value

    def update(self, instance, validated_data):
        if validated_data.get("status") == ChatSession.Status.ARCHIVED:
            validated_data["archived_at"] = timezone.now()
        elif validated_data.get("status") == ChatSession.Status.ACTIVE:
            validated_data["archived_at"] = None

        return super().update(instance, validated_data)


class ChatSessionArchiveSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = ["id", "title", "status", "archived_at"]
        read_only_fields = fields


class ChatSessionListSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = [
            "id",
            "title",
            "mode",
            "status",
            "total_messages",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields
