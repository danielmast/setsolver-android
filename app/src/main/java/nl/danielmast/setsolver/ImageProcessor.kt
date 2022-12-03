package nl.danielmast.setsolver

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc.*
import java.util.stream.Collectors


class ImageProcessor {

    private val TAG = "SetSolver:ImageProcessor"

    private var rgb: Mat? = null
    private var gray: Mat? = null
    private var normalized: Mat? = null
    private var blurred: Mat? = null
    private var thresholded: Mat? = null
    private var drawn: Mat? = null

    private val BLUR_SIZE = 5.0
    private val CARD_THRESHOLD = 140.0
    private val APPROXIMATION_FACTOR = 0.02
    private val MIN_RATIO_OF_MAX_AREA = 0.3
    private val DRAW_LIMIT = 13
    private val CARD_WIDTH = 60.0
    private val CARD_HEIGHT = 90.0

    fun getCards(image: Mat) : Mat {
        if (rgb == null) {
            rgb = Mat()
            gray = Mat()
            normalized = Mat()
            blurred = Mat()
            thresholded = Mat()
            drawn = Mat()
        }

        cvtColor(image, rgb, COLOR_BGR2RGB)
        cvtColor(rgb, gray, COLOR_RGB2GRAY)
        Core.normalize(gray, normalized, 0.0, 255.0, Core.NORM_MINMAX)
        blur(normalized, blurred, Size(BLUR_SIZE, BLUR_SIZE))
        threshold(blurred, thresholded, CARD_THRESHOLD, 255.0, THRESH_BINARY)

        val cardContours = getCardContours(thresholded!!)

        val cardImagesAndPositions: Map<Mat, Point> = cardContours.parallelStream()
            .collect(
                Collectors.toMap(
                    { c -> getCardImage(rgb!!, c) },
                    { c -> getCenter(c) }
                )
            )

        val cardPositions = cardContours.map { c -> getCenter(c) }

        Log.i(TAG, String.format("#contours = %d", cardContours.size))

        image.copyTo(drawn)
        val color = Scalar(220.0, 117.0, 0.0)

        for (i in cardPositions.indices) {
            val card1 = cardPositions[i]
            val card2 = cardPositions[(i + 1) % cardPositions.size]
//            val card = cardImagesAndPositions.values[i]
            line(drawn, card1, card2, color, 4, LINE_AA)
        }

        return drawn!!
    }

    private fun getCardContours(thresholded: Mat): List<MatOfPoint> {
        val contours: List<MatOfPoint> = ArrayList()

        // Get all (external) contours
        findContours(thresholded, contours, Mat(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        // Get all approximated contours
        val approximations: List<MatOfPoint> = contours.map { i -> getApproximation(i) }

        // Get the largest contour area
        val maxArea = approximations
            .map { contour: MatOfPoint ->
                contourArea(
                    contour
                )
            }.maxOrNull() ?: throw IllegalArgumentException("Cannot compute maxArea")
//            .maxOfWith(Comparator<Db>.naturalOrder())
////            .maxOfWithOrNull(Comparator.naturalOrder())
////            .max(Comparator.naturalOrder())
//            .orElseThrow {
//                RuntimeException(
//                    "No approximations"
//                )
//            }

        // Filter the largest contours
        // This should remain only the contours of the cards
        return approximations
            .filter { a: MatOfPoint -> a.height() == 4 } // TODO Dont understand yet
            .filter { a: MatOfPoint -> contourArea(a) > maxArea * MIN_RATIO_OF_MAX_AREA }
    }

    private fun getApproximation(contour: MatOfPoint): MatOfPoint {
        val contour2f = MatOfPoint2f(*contour.toArray())
        approxPolyDP(contour2f, contour2f, arcLength(contour2f, true) * APPROXIMATION_FACTOR, true)
        return MatOfPoint(*contour2f.toArray())
    }

    private fun getCenter(contour: MatOfPoint): Point {
        val m = moments(contour)
        return Point(
            m._m10 / m._m00,
            m._m01 / m._m00
        )
    }

    private fun getCardImage(image: Mat, contour: MatOfPoint): Mat {
        val tempRect: MatOfPoint2f = getOrderedContour(contour)
        val dst = MatOfPoint2f()
        dst.fromArray(
            Point(0.0, 0.0),
            Point(CARD_WIDTH - 1.0, 0.0),
            Point(CARD_WIDTH - 1.0, CARD_HEIGHT - 1.0),
            Point(0.0, CARD_HEIGHT - 1.0)
        )
        val m = getPerspectiveTransform(tempRect, dst)
        val cardImage = Mat()
        warpPerspective(image, cardImage, m, Size(CARD_WIDTH, CARD_HEIGHT))
        return cardImage
    }

    private fun getOrderedContour(contour: MatOfPoint): MatOfPoint2f {
        val boundingRect = boundingRect(contour)
        val w = boundingRect.width
        val h = boundingRect.height
        val sortedByX = sortByX(contour)
        val sortedByY = sortByY(contour)
        val ratio = w.toDouble() / h.toDouble()
        return if (ratio > 1.35 || ratio < 1.0 / 1.35) {
            val minXPlusY = sortedByX.stream()
                .min(Comparator.comparing { p -> p.x + p.y }).get()
            val maxXPlusY = sortedByX.stream()
                .max(Comparator.comparing { p -> p.x + p.y }).get()
            val otherMinX = if (sortedByX[0].equals(minXPlusY)) sortedByX[1] else sortedByX[0]
            val otherMaxX = if (sortedByX[3].equals(maxXPlusY)) sortedByX[2] else sortedByX[3]
            if (w < h) {
                val flavourC = MatOfPoint2f()
                flavourC.fromArray(minXPlusY, otherMaxX, maxXPlusY, otherMinX)
                flavourC
            } else if (w > h) {
                val flavourD = MatOfPoint2f()
                flavourD.fromArray(otherMinX, minXPlusY, otherMaxX, maxXPlusY)
                flavourD
            } else {
                throw RuntimeException("Unexpected dimensions")
            }
        } else {
            val xOfYMin = sortedByY[0].x
            val xOfYMax = sortedByY[3].x
            val yOfXMin = sortedByX[0].y
            val yOfXMax = sortedByX[3].y
            val flavourA = MatOfPoint2f()
            flavourA.fromArray(sortedByX[0], sortedByY[0], sortedByX[3], sortedByY[3])
            val flavourB = MatOfPoint2f()
            flavourB.fromArray(sortedByY[3], sortedByX[0], sortedByY[0], sortedByX[3])
            if (xOfYMin < xOfYMax) {
                if (yOfXMin < yOfXMax) {
                    flavourA
                } else if (yOfXMin > yOfXMax) {
                    if (w < h) {
                        flavourB
                    } else if (w > h) {
                        flavourA
                    } else {
                        throw RuntimeException("Unexpected dimensions")
                    }
                } else {
                    flavourA
                }
            } else if (xOfYMin > xOfYMax) {
                if (yOfXMin < yOfXMax) {
                    if (w < h) {
                        flavourA
                    } else if (w > h) {
                        flavourB
                    } else {
                        throw RuntimeException("Unexpected dimensions")
                    }
                } else if (yOfXMin > yOfXMax) {
                    flavourB
                } else {
                    flavourB
                }
            } else {
                if (yOfXMin < yOfXMax) {
                    flavourA
                } else if (yOfXMin > yOfXMax) {
                    flavourB
                } else {
                    throw RuntimeException("Unexpected dimensions")
                }
            }
        }
    }

    private fun sortByX(pts: MatOfPoint): List<Point> {
        return pts.toList()
            .sortedWith(Comparator.comparing { p -> p.x })
    }

    private fun sortByY(pts: MatOfPoint): List<Point> {
        return pts.toList()
            .sortedWith(Comparator.comparing { p -> p.y })
    }
}