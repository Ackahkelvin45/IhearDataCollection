from storages.backends.s3boto3 import S3Boto3Storage

class StaticStorage(S3Boto3Storage):
    location = 'static'
    default_acl = 'public-read'
    file_overwrite = True
    custom_domain = 'datacollectionfiles.lon1.cdn.digitaloceanspaces.com'
    
    def path(self, name):
        return name  # Minimal implementation to satisfy Django