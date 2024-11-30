
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from document_upload.models import Book


class UserBooksView(APIView):
    def get(self, request, *args, **kwargs):
        user_id = request.query_params.get('user_id')
        print(user_id)
        if not user_id:
            return Response({"error": "User ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        books = Book.objects.filter(user=user_id)
        book_list = [
            {
                "name": book.name,
                "url": book.url,
                "uploaded_at": book.uploaded_at
            } for book in books
        ]

        return Response(book_list, status=status.HTTP_200_OK)
