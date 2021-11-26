package com.example.opencl;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import com.example.opencl.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'opencl' library on application startup.
    static {
        System.loadLibrary("OpenCLDriver");
    }

    // Blur CPU
    public native Bitmap GaussianBlurBitmap(Bitmap bitmap);

    // Blur GPU
    public native Bitmap GaussianBlurGPU(Bitmap bitmap);

    // ViewBinding
    private ActivityMainBinding binding;

    public Bitmap buf_bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;

        buf_bitmap = BitmapFactory.decodeFile("/data/local/tmp/lena.bmp", options);
        binding.tvTimeVal.setText("Image Processing App");
        binding.ivMain.setImageBitmap(buf_bitmap);

        binding.btnCpuBlur.setOnClickListener(v -> {
            float start = (float) System.nanoTime() / 1000000;

            Bitmap bitmapCPU = GaussianBlurBitmap(buf_bitmap);

            float end = (float) System.nanoTime() / 1000000;
            float timeSub = end - start;

            binding.ivMain.setImageBitmap(bitmapCPU);
            binding.tvTimeVal.setText("Execution Time : " + timeSub);
        });

        binding.btnGpuBlur.setOnClickListener(v -> {
            float start = (float) System.nanoTime() / 1000000;

            Bitmap bitmapGPU = GaussianBlurGPU(buf_bitmap);

            float end = (float) System.nanoTime() / 1000000;
            float timeSub = end - start;

            binding.ivMain.setImageBitmap(bitmapGPU);
            binding.tvTimeVal.setText("Execution Time : " + timeSub);
        });

        binding.btnOriginal.setOnClickListener(v -> {
            BitmapFactory.Options options1 = new BitmapFactory.Options();
            options1.inPreferredConfig = Bitmap.Config.ARGB_8888;
            buf_bitmap =
                    BitmapFactory.decodeFile("/data/local/tmp/lena.bmp",
                            options1);
            binding.ivMain.setImageBitmap(buf_bitmap);
            binding.tvTimeVal.setText("Orignal image");
        });
    }
}