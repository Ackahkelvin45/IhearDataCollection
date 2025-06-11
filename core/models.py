from django.db import models

# Create your models here.



class Region(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name
    
class Category(models.Model):
    name = models.CharField(max_length=100, unique=True, null=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name


class Class(models.Model):
    name = models.CharField(max_length=100, null=True)
    category = models.ForeignKey(
        Category, 
        on_delete=models.PROTECT,
        related_name='classes',

         null=True
    )
    description = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = ('name', 'category')  # Ensure class names are unique within a category

    def __str__(self):
        return f"{self.category.name} - {self.name}"


class SubClass(models.Model):
    name = models.CharField(max_length=100)
    parent_class = models.ForeignKey(
        Class, 
        on_delete=models.PROTECT,
        related_name='subclasses',
        null=True
    )
    description = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = ('name', 'parent_class')  # Ensure subclass names are unique within a class

    def __str__(self):
        return f"{self.parent_class} - {self.name}"
    def __str__(self):
        return self.name
    
class Community(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name
    

class Microphone_Type(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name
    


    
class Time_Of_Day(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name


    




