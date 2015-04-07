package com.xuey90.regie_video;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.media.FaceDetector;
import android.media.FaceDetector.Face;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

public class AndroidFaceDetection extends Activity {
    private static final int SELECT_GALLERY_CODE = 1;
    private static final int TAKE_PICTURE_CODE = 2;

    Button bTakePic, bSelectPic, bSend;
    EditText mEdit;
    static ImageView imgView;
    // private MyImageView imgView;
    public int imageWidth;
    public int imageHeight;
    public int numberOfFace = 5;
    public FaceDetector myFaceDetect;
    public FaceDetector.Face[] myFace;
    float myEyesDistance;
    int numberOfFaceDetected;
    Bitmap mFaceBitmap = null;
    InputStream stream;
    int[] fpx = null;
    int[] fpy = null;
    Canvas canvas = new Canvas();
    Paint myPaint = new Paint();
    ContentValues values;
    Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // TODO Auto-generated method stub
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        bTakePic = (Button) findViewById(R.id.takePicture);
        bSelectPic = (Button) findViewById(R.id.selectPicture);
        bSend = (Button)findViewById(R.id.button_send);
        mEdit = (EditText)findViewById(R.id.edit_message);
        imgView = (ImageView) findViewById(R.id.imageView1);

        bSelectPic.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                startActivityForResult(intent, SELECT_GALLERY_CODE);
            }
        });
        bTakePic.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                values = new ContentValues();
                values.put(MediaStore.Images.Media.TITLE, "New Picture");
                values.put(MediaStore.Images.Media.DESCRIPTION, "From your Camera");
                imageUri = getContentResolver().insert(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);

                startActivityForResult(intent, TAKE_PICTURE_CODE);
            }
        });

        bSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String name = mEdit.getText().toString();
                if(mFaceBitmap == null){
                    Toast toast =  Toast.makeText(AndroidFaceDetection.this, "No image found! Please add a image!", Toast.LENGTH_LONG);
                    toast.setGravity(Gravity.CENTER,0,0);
                    toast.show();
                }
                else if(name.isEmpty()){
                    Toast.makeText(AndroidFaceDetection.this, "Name Empty! Please input a name!", Toast.LENGTH_LONG).show();
                }
                else {
                    Toast.makeText(AndroidFaceDetection.this,"Image of " + name + " sent to Robot!", Toast.LENGTH_LONG).show();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == SELECT_GALLERY_CODE
                && resultCode == Activity.RESULT_OK) {
            try {
                // We need to recyle unused bitmaps
                if (mFaceBitmap != null) {
                    mFaceBitmap.recycle();
                }

                stream = getContentResolver().openInputStream(data.getData());
                Bitmap b = BitmapFactory.decodeStream(stream);

                mFaceBitmap = b.copy(Bitmap.Config.RGB_565, true);
                b.recycle();
                imgView.setImageBitmap(mFaceBitmap);
                DetectFaces();
                stream.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else if (TAKE_PICTURE_CODE == requestCode)
            if (resultCode == Activity.RESULT_OK) {

                try {
                    Bitmap b = MediaStore.Images.Media.getBitmap(
                            getContentResolver(), imageUri);
                    mFaceBitmap = b.copy(Bitmap.Config.RGB_565, true);
                    b.recycle();
                    imgView.setImageBitmap(mFaceBitmap);
                    DetectFaces();
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        super.onActivityResult(requestCode, resultCode, data);
    }


    public void DetectFaces() {
        // TODO Auto-generated method stub

        imageWidth = mFaceBitmap.getWidth();
        imageHeight = mFaceBitmap.getHeight();
        imgView.setImageBitmap(mFaceBitmap);

        myFace = new FaceDetector.Face[numberOfFace];
        myFaceDetect = new FaceDetector(imageWidth, imageHeight, numberOfFace);
        numberOfFaceDetected = myFaceDetect.findFaces(mFaceBitmap, myFace);
        //Toast.makeText(AndroidFaceDetection.this, "No of Face Detected: "+numberOfFaceDetected, Toast.LENGTH_LONG).show();
        Paint ditherPaint = new Paint();
        ditherPaint.setDither(true);

        myPaint.setColor(Color.GREEN);
        myPaint.setStyle(Paint.Style.STROKE);
        myPaint.setStrokeWidth(3);

        canvas.setBitmap(mFaceBitmap);
        canvas.drawBitmap(mFaceBitmap, 0, 0, ditherPaint);

        if (numberOfFaceDetected > 0) {
            for (int i = 0; i < numberOfFaceDetected; i++) {
                Face face = myFace[i];
                PointF myMidPoint = new PointF();
                face.getMidPoint(myMidPoint);

                myEyesDistance = face.eyesDistance();

                Toast.makeText(AndroidFaceDetection.this, "Eye distance: "+myEyesDistance, Toast.LENGTH_LONG).show();
                canvas.drawRect((int) (myMidPoint.x - myEyesDistance ),
                        (int) (myMidPoint.y - myEyesDistance ),
                        (int) (myMidPoint.x + myEyesDistance ),
                        (int) (myMidPoint.y + myEyesDistance * 1.5), myPaint);


            }

        }

    }

}
