package nl.danielmast.setsolver

import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat


class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener {

    private val TAG = "SetSolver:MainActivity"

    private val imageProcessor = ImageProcessor()

    private lateinit var mOpenCvCameraView: CameraBridgeViewBase

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            Log.i(TAG, "onManagerConnected")
            when (status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        mOpenCvCameraView = findViewById(R.id.OpenCvView)
        mOpenCvCameraView.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)
    }

    override fun onPause() {
        Log.i(TAG, "onPause")
        super.onPause()

        mOpenCvCameraView.disableView()
    }

    override fun onResume() {
        Log.i(TAG, "onResume")
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(
                TAG,
                "Internal OpenCV library not found. Using OpenCV Manager for initialization"
            )
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(
                TAG,
                "OpenCV library found inside package. Using it!"
            )
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        super.onDestroy()
        mOpenCvCameraView.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.i(TAG, "onCameraViewStarted")
    }

    override fun onCameraViewStopped() {
        Log.i(TAG, "onCameraViewStopped")
    }

    override fun onCameraFrame(inputFrame: Mat): Mat {
        return try {
            imageProcessor.getCards(inputFrame)
        } catch (e: Exception) {
            Log.e(TAG,"Failed to get cards", e)
            inputFrame
        }
    }
}